import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utilis.utils import setupLogger
logger = setupLogger(__name__)

class GonStrategyContract:
    '''
    this is the contract of the strategy.
    the strategy holds USDT of investors and assuming a certain daily retuns which determines
    the TVL of the strategy.
    it also holds the balance of the liquidity providers
    '''
    def __init__(self, strategy_name, reward_fee_on_pnl_bps):
        self._strategy_name = strategy_name
        # dict of gon_participant_name: amount_usdt
        # the amount_usdt represent the postion of their TVL in the strategy, influenced by pnl daily
        self._gon_participants = {}

        # dict of gon_participant_name: amount_gon
        # the amount_gon represent the amount of GON tokens the investor holds
        self._gon_rewards = {}

        self._reward_fee_on_pnl_bps = reward_fee_on_pnl_bps

        self._index_price = 100.

    def __str__(self):
        return f'{self._strategy_name}'

    def get_index_price(self):
        return self._index_price

    def get_strategy_tvl(self):
        return sum(self._gon_participants.values())

    def get_gon_tvl(self):
        return sum(self._gon_rewards.values())

    def get_gon_participants(self):
        return self._gon_participants

    def get_gon_participants_rewards(self):
        return self._gon_rewards

    def deposit(self, amount_usdt, gon_participant_name):
        '''
        deposit USDT to the strategy
        :param amount_usdt: amount of USDT to deposit
        :param gon_participant_name: name of the investor
        :return: self
        '''
        assert amount_usdt > 0
        if self._gon_participants.get(gon_participant_name) is None:
            self._gon_participants[gon_participant_name] = amount_usdt
        else:
            self._gon_participants[gon_participant_name] += amount_usdt
        return amount_usdt

    def withdraw(self, amount_usdt, gon_participant_name):
        '''
        withdraw USDT from the strategy
        :param amount_usdt: amount of USDT to withdraw
        :param gon_participant_name: name of the investor
        :return: self
        '''
        assert amount_usdt > 0
        gon_investor_balance = self._gon_participants.get(gon_participant_name)
        if gon_investor_balance < amount_usdt:
            raise ValueError(f'gon_invertor_balance < amount_usdt: {gon_investor_balance} < {amount_usdt}')

        self._gon_participants[gon_participant_name] -= amount_usdt
        return amount_usdt

    def withdraw_all(self, gon_participant_name):
        '''
        withdraw all USDT from the strategy for the gon_participant_name
        :param gon_participant_name: name of the investor
        :return: self
        '''
        gon_investor_balance = self._gon_participants.get(gon_participant_name, 0.)

        self._gon_participants[gon_participant_name] = 0.0
        return gon_investor_balance

    def withdraw_rewards(self, gon_participant_name):
        '''
        withdraw all GON rewards from the strategy for the gon_participant_name
        :param gon_participant_name: name of the investor
        :return: self
        '''
        gon_investor_rewards_balance = self._gon_rewards.get(gon_participant_name, 0.)

        self._gon_rewards[gon_participant_name] = 0.0
        return gon_investor_rewards_balance

    def compute_performance_fee(self, pnl_usdt):
        '''
        compute the performance fee of the strategy
        :param pnl_usdt: the pnl of the strategy
        :return: performance_fee_usdt: the performance fee of the strategy
        '''
        performance_fee_usdt = pnl_usdt * self._reward_fee_on_pnl_bps / 10000
        pnl_usdt_adjusted = pnl_usdt - performance_fee_usdt
        return performance_fee_usdt, pnl_usdt_adjusted

    def update_tvl(self, pnl_usdt):
        '''
        update the TVL of the strategy
        :param pnl_usdt: the pnl of the strategy
        :return: self
        '''

        if pnl_usdt > 0:
            logger.debug(f'pnl_usdt: {pnl_usdt} > 0, computing performance fee...')
            performance_fee_usdt, pnl_usdt_adjusted = self.compute_performance_fee(pnl_usdt)
            logger.debug(f'performance_fee_usdt: {performance_fee_usdt}')
        else:
            performance_fee_usdt = 0.0 # no performance fee
            pnl_usdt_adjusted = pnl_usdt # no performance fee

        self._index_price = self._index_price * (1 + pnl_usdt_adjusted / self.get_strategy_tvl())
        self._update_gon_participant_balance(pnl_usdt=pnl_usdt_adjusted)

        strategy_tvl = sum(self._gon_participants.values())
        assert strategy_tvl >= 0
        return performance_fee_usdt

    # def _update_gon_participant_rewards(self, rewards_usdt):
    #     '''
    #     given the strategy TVL, update the balance of the gon participants
    #     :return: self
    #     '''
    #     for gon_participant_name, gon_participant_balance in self._gon_participants.items():
    #         pct_of_pool = self._gon_participants[gon_participant_name]/self.get_strategy_tvl()
    #         zero_or_update = max( pct_of_pool*rewards_usdt + self._gon_participants[gon_participant_name],0)
    #         if zero_or_update == 0:
    #             raise ValueError(f'zero_or_update == 0: {zero_or_update}')
    #         self._gon_participants[gon_participant_name] = zero_or_update

    def _update_gon_participant_balance(self, pnl_usdt):
        '''
        given the strategy TVL, update the balance of the gon participants
        :return: self
        '''
        strategy_tvl = self.get_strategy_tvl()
        for gon_participant_name, gon_participant_balance in self._gon_participants.items():
            pct_of_pool = self._gon_participants[gon_participant_name]/strategy_tvl
            zero_or_update = max( pct_of_pool*pnl_usdt + self._gon_participants[gon_participant_name],0)
            #if zero_or_update == 0:
                #logger.warning(f'zero_or_update == 0: {zero_or_update}, gon_participant_name: {gon_participant_name}')
                # raise ValueError(f'zero_or_update == 0: {zero_or_update}')
            self._gon_participants[gon_participant_name] = zero_or_update

    def compute_paticipants_pct(self):
        '''
        compute the pool pct of the strategy
        :return: pool_pct: the pool pct of the strategy
        '''
        participants_pct = pd.Series(self._gon_participants) / self.get_strategy_tvl()
        return participants_pct

    def allocate_gonomics_rewards(self, gon_rewards):
        '''
        allocate gon rewards to the gon participants
        :param gon_rewards: the gon rewards to allocate
        :return: self
        '''
        for gon_participant_name, gon_participant_balance in self._gon_participants.items():
            self._gon_rewards[gon_participant_name] = self._gon_rewards.get(gon_participant_name, 0.0) + gon_rewards.loc[gon_participant_name]
        return self
