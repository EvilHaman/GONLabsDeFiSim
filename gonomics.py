import pandas as pd
import warnings

from utilis.constants import GON_PARTICIPANT_ACTION_TYPE, GON_PARTICIPANT_TYPE
from gon_trades_recorder import GonTransactions

warnings.simplefilter(action='ignore', category=FutureWarning)
from utilis.utils import setupLogger
logger = setupLogger(__name__)

class GonDistributionScheduler:
    '''
    this class will schedule the distribution of GON tokens to various actors in the ecosystem.
    '''
    def __init__(self, GON_distribution, GON_distribution_schedule, datetime_first):
        '''
        :param GON_distribution: pd.Series of the name_of_gon_holder_type to its % contribution:
        :param GON_distribution_schedule: a name_of_gon_holder_type to its % contribution.
               Note that the first and last months for all gon members must have a number.
               this class by default will linearly interpolate between the missing months
        :param datetime_first: datetime of the first month of the distribution schedule
        '''
        self._datetime_first = datetime_first

        self._GON_distribution = GON_distribution

        ## assert the distribution add to 100% of the supply:
        assert GON_distribution.sum() == 1.

        ## assert the structure of the distribution schedule must start with a value and finish with a value for the same month for all gon members:
        assert (not GON_distribution_schedule.iloc[0].isnull().any()) and\
               (not GON_distribution_schedule.iloc[-1].isnull().any()) and\
               (GON_distribution_schedule.iloc[-1] == 1.).all()

        ## interpolate the GON_distribution_schedule
        self._GON_distribution_schedule = pd.Series([i for i in range(GON_distribution_schedule.idxmax().max() + 1)], name='month') \
            .to_frame().merge(GON_distribution_schedule, how='left', left_on='month', right_index=True) \
            .interpolate(method='linear').set_index('month')\
        .assign(TOTAL = lambda x: (x * self._GON_distribution).sum(axis=1))\
        .assign(datetime = lambda x: x.index.map(lambda y: self._datetime_first + pd.DateOffset(months=y)) )

    def compute_total_rewards_today(self, datetime, gon_total_supply):
        '''
        compute the total rewards to be distributed today
        :param datetime: datetime of today
        :param gon_total_supply: total supply of GON tokens
        :return: total rewards to be distributed today
        '''
        #gon_total_supply=200*1e7
        this_month_distribution = self._GON_distribution_schedule.where( (self._GON_distribution_schedule['datetime'].dt.month == datetime.month) \
                                              & (self._GON_distribution_schedule['datetime'].dt.year == datetime.year) )\
        .dropna()
        if len(this_month_distribution) == 0:
            logger.warning(f'no distribution for {datetime}, will not distribute any GON tokens...')
            return pd.DataFrame()

        else:
            number_of_days_this_month = this_month_distribution['datetime'].dt.days_in_month.values[0]
            today_gon_to_distribution_today = gon_total_supply * this_month_distribution['TOTAL'].values[0] / number_of_days_this_month
            gon_daily_distribution = today_gon_to_distribution_today * self._GON_distribution * this_month_distribution.drop(['TOTAL', 'datetime'], axis=1).iloc[0]/number_of_days_this_month

            return gon_daily_distribution


class GonomicsEngine:
    '''
    this is the GONLabs tokenomics engine.
    it's goals are:
    1. distribute GON tokens to various actors, based on a mechanism
    2. buy & burn GON tokens from the market
    3. keep track of the GON token distribution
    '''
    def __init__(self, gon_total_supply=200*1e7, gas_fee_usdt=0.):
        '''
        :param gon_total_supply: total supply of GON tokens
        '''
        self._GON_TOTAL_SUPPLY = gon_total_supply
        self._gon_supply_remaining = gon_total_supply
        self._gon_transactions = GonTransactions()
        self.GAS_FEE_USDT = gas_fee_usdt

    def set_distribution_schedule(self, GON_distribution, GON_distribution_schedule, datetime_first):
        '''
        set the distribution schedule of GON tokens
        :param GON_distribution: pd.Series of the name_of_gon_holder_type to its % contribution:
        :param GON_distribution_schedule: a name_of_gon_holder_type to its % contribution.
               Note that the first and last months for all gon members must have a number.
               this by default will linearly interpolate between the missing months
        :param datetime_first: datetime of the first month of the distribution schedule
        '''
        self._gon_distribution_scheduler = GonDistributionScheduler(GON_distribution=GON_distribution,
                                                                    GON_distribution_schedule=GON_distribution_schedule,
                                                                    datetime_first=datetime_first)
        return self

    def __str__(self):
        return f'GonomicsEngine'

    def get_gon_total_supply(self):
        return self._GON_TOTAL_SUPPLY

    def buy_and_burn(self, amount_usdt, amm_pool, gas_fee_usdt=None, datetime=None):
        '''
        buy GON tokens from the market and burn them
        :param amount_usdt: amount of USDT to buy GON tokens with
        :param amm_pool: AMM pool of the GON-USDT pair
        :return: amount of GON tokens bought
        '''
        gon_burned = amm_pool.swapCoin1ToCoin0(amount1_in=amount_usdt,
                                  amount0_out_min=0, to='0x00')
        self._gon_transactions.recordTrade(symbol='GON', symbol_pair='GON/USDT',
                                           side='BUY', amount=gon_burned,
                                           price=amount_usdt/gon_burned,
                                           datetime=datetime,
                                           gas_fee_usdt=self.GAS_FEE_USDT if \
                                               gas_fee_usdt is None else gas_fee_usdt,
                                           action_type=GON_PARTICIPANT_ACTION_TYPE.BUY_AND_BURN)
        return gon_burned

    # def distribute_gon_rewards_on_strategy(self, strat_contract, datetime):
    #     '''
    #     distribute GON tokens to the participants of the strategy
    #     :param strat_contract: the strategy contract
    #     :param datetime: datetime of the distribution. this will be used to calculate the amount of GON tokens to distribute
    #     NOTE: this will excplicitly distribute the GON rewards to the GON_PARTICIPANT_TYPE.COMMUNITY
    #     :return: self
    #     '''
    #     gon_daily_distribution = self._gon_distribution_scheduler.compute_total_rewards_today(datetime=pd.to_datetime(datetime), gon_total_supply=self._GON_TOTAL_SUPPLY)
    #
    #     ## note this will happen in case where the distribution schedule is already completed.
    #     if len(gon_daily_distribution) == 0.:
    #         return {}
    #
    #     gon_rewards = gon_daily_distribution.loc[GON_PARTICIPANT_TYPE.COMMUNITY] * strat_contract.compute_paticipants_pct()
    #     strat_contract.allocate_gonomics_rewards(gon_rewards)
    #     return gon_rewards

    def distribute_gon_rewards(self, strat_contract, amm_pool, datetime):
        '''
        distribute GON tokens to the participants of the strategy
        :param strat_contract: the strategy contract
        :param amm_pool: AMM pool of the GON-USDT pair
        :param datetime: datetime of the distribution. this will be used to calculate the amount of GON tokens to distribute
        :return: self
        '''
        gon_daily_distribution = self._gon_distribution_scheduler.compute_total_rewards_today(datetime=pd.to_datetime(datetime), gon_total_supply=self._GON_TOTAL_SUPPLY)

        ## note this will happen in case where the distribution schedule is already completed.
        if len(gon_daily_distribution) == 0.:
            return ()

        ## distribute to strategy participants
        strategy_gon_rewards = gon_daily_distribution.loc[GON_PARTICIPANT_TYPE.COMMUNITY] * strat_contract.compute_paticipants_pct()
        strat_contract.allocate_gonomics_rewards(strategy_gon_rewards)

        ## distribute to LPs
        gon_rewards_lps = gon_daily_distribution.loc[GON_PARTICIPANT_TYPE.LP_PROVIDER] * amm_pool.compute_paticipants_pct()
        liquidity_token_rewards, amm_to_usdt = amm_pool.allocate_gonomics_rewards_as_liquidity_tokens(gon_rewards_lps=gon_rewards_lps)
        amm_rewards = gon_rewards_lps.rename('gon').to_frame().merge(liquidity_token_rewards.rename('liquidity_token'), left_index=True, right_index=True)\
        .assign(amm_to_usdt=amm_to_usdt)

        ## distribute to founders
        gon_rewards_founders = gon_daily_distribution.loc[GON_PARTICIPANT_TYPE.FOUNDING_TEAM]

        ## distribute to pre-seeders
        gon_rewards_pre_seeders = gon_daily_distribution.loc[GON_PARTICIPANT_TYPE.PRE_SEEDER]

        return (strategy_gon_rewards, amm_rewards, gon_rewards_founders, gon_rewards_pre_seeders)

