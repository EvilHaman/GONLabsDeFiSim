import pandas as pd
import random, string
import numpy as np
from utilis.utils import setupLogger
import warnings

from utilis.constants import GON_PARTICIPANT_TYPE, GON_PARTICIPANT_ACTION_TYPE
from gon_trades_recorder import GonTransactions

warnings.simplefilter(action='ignore', category=FutureWarning)
logger = setupLogger(__name__)


class GonParticipant:
    '''
    participant (investor, LP provider, founders, ect..) which holds USDT invest in GON strategies or other mechanisms in the ecosystem.
    for that he gets rebates of GON coins.
    the investor can deposit GON coins in AMM to earn fees, or to borrow more USDT and invest again in the strategies
    '''
    def __init__(self, name=None, type=GON_PARTICIPANT_TYPE.COMMUNITY,
                 GAS_FEE_USDT = 15.0):
        self._USDT = 0.0
        self._GON = 0.0
        self._gon_transactions = GonTransactions()
        self._stats = {}#pd.DataFrame(columns=['datetime', 'strat_pnl_usdt', 'strat_rtn'])
        self._analytics = []

        if name is None:
            ## generate a random name:
            self._name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        else:
            self._name = name
        self._type = type

        ## gas fee for each transaction
        self.GAS_FEE_USDT = GAS_FEE_USDT

    def __str__(self):
        return f'{self._type}-{self._name}'

    def get_investor_name(self):
        return self._name

    def deposit(self, amount_usdt, gas_fee_usdt=None, datetime=None):
        self._USDT += amount_usdt
        self._record_transaction(symbol='USDT', symbol_pair='USDT/USDT',
                                    side='BUY', amount=amount_usdt,
                                    price=1.0, datetime=datetime,
                                    gas_fee_usdt=gas_fee_usdt if gas_fee_usdt is not None else self.GAS_FEE_USDT,
                                    action_type=GON_PARTICIPANT_ACTION_TYPE.DEPOSIT)
        return self

    def withdraw(self, amount_usdt):
        self._USDT -= amount_usdt
        return amount_usdt

    def _compute_investment_fee_adjusted(self, pct_of_total, account='USDT', **kwargs):
        if account == 'USDT':
            investment_amount_usdt = self._USDT * pct_of_total
        else:
            gon_to_usdt_price = kwargs.get('gon_to_usdt_price', None)
            if gon_to_usdt_price is None:
                raise ValueError('gon_to_usdt_price not provided while trying to compute investment with GON')
            investment_amount_usdt = self._GON * pct_of_total * gon_to_usdt_price


        if investment_amount_usdt < self.GAS_FEE_USDT:
            logger.warning(f'investment_amount_usdt < GAS_FEE_USDT for {self._name}, will not take money out of account...')
            return 0
        if investment_amount_usdt < 0:
            raise ValueError('investment_amount_usdt < 0')
        elif investment_amount_usdt == 0:
            logger.warning(f'investment_amount_usdt == 0 for {self._name}, will not take money out of account...')
            return 0

        if account == 'USDT':
            self._USDT = self._USDT - investment_amount_usdt
        else:
            self._GON = self._GON - investment_amount_usdt/gon_to_usdt_price

        investment_amount_usdt_gas_adjusted = investment_amount_usdt - self.GAS_FEE_USDT
        return investment_amount_usdt_gas_adjusted

    def invest_in_strategy(self, pct_of_total_usdt, strat_contract, datetime=None):
        investment_amount_usdt_gas_adjusted = self._compute_investment_fee_adjusted(pct_of_total=pct_of_total_usdt)

        if investment_amount_usdt_gas_adjusted > 0:
            strat_contract.deposit(amount_usdt=investment_amount_usdt_gas_adjusted,
                                   gon_participant_name=self.get_investor_name())

        self._record_transaction(symbol='USDT', symbol_pair='STRATEGY/USDT',
                                    side='BUY', amount=investment_amount_usdt_gas_adjusted/strat_contract.get_index_price(),
                                    price=strat_contract.get_index_price(), datetime=datetime,
                                    action_type=GON_PARTICIPANT_ACTION_TYPE.INVEST_IN_STRATEGY)
        return investment_amount_usdt_gas_adjusted

    def unvest_in_strategy(self, strat_contract, gon_to_usdt, datetime):
        usdt_amount = strat_contract.withdraw_all(gon_participant_name=self.get_investor_name())
        self._USDT += usdt_amount
        gon_amount = self.withdraw_rewards_from_strategy(strat_contract=strat_contract,
                                                         gon_to_usdt = gon_to_usdt,
                                                         datetime=datetime)

        if abs(usdt_amount) > 1e-9:
            self._record_transaction(symbol='USDT', symbol_pair='STRATEGY/USDT',
                                        side='SELL', amount=usdt_amount/strat_contract.get_index_price(),
                                        price=strat_contract.get_index_price(), datetime=datetime,
                                        action_type=GON_PARTICIPANT_ACTION_TYPE.UNVEST_IN_STRATEGY)
            return usdt_amount, gon_amount

        else:
            # logger.debug(f'usdt_amount is smaller than 1e9 and = {usdt_amount}, '
            #              f'assume no investment in strategy... ABORT!')
            return usdt_amount, gon_amount


    def swap(self, pct_of_total, amm_pool, datetime, account='USDT'):
        '''
        swap USDT for GON or vice versa in AMM pool
        :param amm_pool: amm pool object
        :return: self
        '''
        gon_to_usdt_price = amm_pool.reserve1 / amm_pool.reserve0
        investment_amount_usdt_gas_adjusted = self._compute_investment_fee_adjusted(pct_of_total=pct_of_total, account=account,
                                                                                    gon_to_usdt_price=gon_to_usdt_price)
        if investment_amount_usdt_gas_adjusted == 0:
            return self

        if account == 'USDT':
            amount_gon = amm_pool.swapCoin1ToCoin0(amount1_in=investment_amount_usdt_gas_adjusted,
                                                   amount0_out_min=0, to=self.get_investor_name())
            self._GON += amount_gon
            self._gon_transactions.recordTrade(symbol='GON', symbol_pair='GON/USDT',
                                               side='BUY', amount=amount_gon,
                                               price=investment_amount_usdt_gas_adjusted / amount_gon,
                                               datetime=datetime,
                                               gas_fee_usdt=self.GAS_FEE_USDT,
                                               action_type=GON_PARTICIPANT_ACTION_TYPE.SWAP)
        else:
            amount_usdt = amm_pool.swapCoin0ToCoin1(amount0_in=investment_amount_usdt_gas_adjusted/ gon_to_usdt_price,
                                                    amount1_out_min=0, to=self.get_investor_name())
            self._USDT += amount_usdt
            self._gon_transactions.recordTrade(symbol='GON', symbol_pair='GON/USDT',
                                               side='SELL', amount=investment_amount_usdt_gas_adjusted/ gon_to_usdt_price,
                                               price=amount_usdt / (investment_amount_usdt_gas_adjusted/ gon_to_usdt_price),
                                               datetime=datetime,
                                               gas_fee_usdt=self.GAS_FEE_USDT,
                                               action_type=GON_PARTICIPANT_ACTION_TYPE.SWAP)
        return self

    def record_reward(self, amount, datetime, symbol='GON', symbol_pair='STRATEGY/GON', action_type=GON_PARTICIPANT_ACTION_TYPE.RECORD_REWARDS_ON_STRATEGY):
        self._record_transaction(symbol=symbol, symbol_pair=symbol_pair,
                                 side='BUY', amount=amount,
                                 price=0., datetime=datetime,
                                 action_type=action_type)
        return True

    def withdraw_rewards_from_strategy(self, strat_contract, gon_to_usdt, datetime):
        gon_amount = strat_contract.withdraw_rewards(gon_participant_name=self.get_investor_name())
        if abs(gon_amount) < 1e-9:
            # logger.debug(f'rewards are smaller than 1e9 and = {gon_amount}, '
            #              f'assume no rewards to withdraw from strategy... ABORT!')
            return gon_amount

        self._GON += gon_amount

        self._record_transaction(symbol='GON', symbol_pair='STRATEGY/GON',
                                    side='SELL', amount=gon_amount,
                                    price=gon_to_usdt, datetime=datetime,
                                    action_type=GON_PARTICIPANT_ACTION_TYPE.WITHDRAW_REWARDS_FROM_STRATEGY)
        return gon_amount

    def _compute_amm_liquidity_split(self, investment_amount_usdt, amm_pool):
        if amm_pool.reserve1 ==0. and amm_pool.reserve0 == 0.:
            logger.warning('reserves of amm are empty, assuming all USDT against given GON tokens in the account...')
            gon_portion = self._GON
            usdt_portion = investment_amount_usdt
            self._GON = 0.
            return gon_portion, usdt_portion

        gon_to_usdt_price = amm_pool.reserve1 / amm_pool.reserve0
        gon_proportion = investment_amount_usdt * .5 * 1 / gon_to_usdt_price
        usdt_proportion = investment_amount_usdt - gon_proportion * gon_to_usdt_price
        assert usdt_proportion * 1e6 - gon_proportion * 1e6 * gon_to_usdt_price < 1e6
        return gon_proportion, usdt_proportion

    def add_amm_liquidity(self, amm_pool, pct_of_total, datetime=None):
        from_account = 'USDT'
        ## NOTE: gon_to_usdt_price is used in case from_account is GON given all computed in $...
        gon_to_usdt_price = amm_pool.reserve1 / amm_pool.reserve0 if amm_pool.reserve0 > 0. else None
        investment_amount_usdt_gas_adjusted = self._compute_investment_fee_adjusted(pct_of_total=pct_of_total,
                                                                                    account=from_account,
                                                                                    gon_to_usdt_price=gon_to_usdt_price)
        if investment_amount_usdt_gas_adjusted > 0:
            gon_proportion, usdt_proportion = self._compute_amm_liquidity_split(investment_amount_usdt=investment_amount_usdt_gas_adjusted,
                                                                                amm_pool=amm_pool)
            lp_balance_before = amm_pool.liquidity_providers.get(self.get_investor_name(), 0.)
            g_amount, u_amount = amm_pool.add_liquidity(_from=self.get_investor_name(), balance1=usdt_proportion,
                                   balance0=gon_proportion,
                                   balance0Min=0, balance1Min=0)
            lp_balance_added = amm_pool.liquidity_providers[self.get_investor_name()] - lp_balance_before

            self._record_transaction(symbol=from_account, symbol_pair='AMM/' + from_account,
                                     side='BUY', amount=lp_balance_added,
                                     price=investment_amount_usdt_gas_adjusted/lp_balance_added, datetime=datetime,
                                     action_type=GON_PARTICIPANT_ACTION_TYPE.ADD_AMM_LIQUIDITY)
            return investment_amount_usdt_gas_adjusted

    def withdraw_amm_liquidity(self, amm_pool, datetime=None):
        lp_balance_to_sell = amm_pool.liquidity_providers.get(self.get_investor_name(), 0.)
        gon_to_usdt = amm_pool.reserve1 / amm_pool.reserve0
        if  lp_balance_to_sell > 0.:
            gon_amount, usdt_amount = amm_pool.remove_liquidity(to=self.get_investor_name(),
                                                                liquidity=amm_pool.liquidity_providers[
                                                                    self.get_investor_name()],
                                                                amount0_min=0, amount1_min=0)

            investment_amount_usdt_unwounded = usdt_amount + gon_amount*gon_to_usdt
            self._GON += gon_amount
            self._USDT += usdt_amount

            self._record_transaction(symbol='USDT', symbol_pair='AMM/USDT',
                                     side='SELL', amount=lp_balance_to_sell,
                                     price=investment_amount_usdt_unwounded/ lp_balance_to_sell , datetime=datetime,
                                     action_type=GON_PARTICIPANT_ACTION_TYPE.REMOVE_AMM_LIQUDITY)
            return gon_amount, usdt_amount

        else:
            # logger.debug('no liquidity to withdraw from amm pool...for account: %s' % self.get_investor_name())
            return 0., 0.


    def _record_transaction(self, symbol, symbol_pair,
                           side, amount, price,
                           datetime,
                           action_type=GON_PARTICIPANT_ACTION_TYPE.INVEST_IN_STRATEGY,
                           gas_fee_usdt=None):
        self._gon_transactions.recordTrade(symbol, symbol_pair,
                                           side, amount,
                                           price, datetime,
                                           gas_fee_usdt=self.GAS_FEE_USDT if \
                                               gas_fee_usdt is None else gas_fee_usdt,
                                           action_type=action_type)

    def gather_analytics(self, datetime, strat_contract, amm_pool):
        '''
        another attempt to compute realised and unrealised pnl on money deployed in both strat_contract and amm_pool
        :param datetime:
        :param strat_contract:
        :param amm_pool:
        :return:
        '''
        if amm_pool.liquidity_providers.get(self.get_investor_name(), 0.) > 0.:
            gon_amount, usdt_amount = amm_pool.check_liquidity(to=self.get_investor_name(),
                                     liquidity=amm_pool.liquidity_providers[self.get_investor_name()],
                                     amount0_min=0, amount1_min=0)
        else:
            gon_amount = 0.
            usdt_amount = 0.

        res = {
            'datetime': datetime,
            'USDT': self._USDT, #USDT in account
            'GON': self._GON, #USD in account
            'strat_USDT': strat_contract.get_gon_participants().get(self.get_investor_name(), 0.), # USDT in strat contract
            'strat_GON': strat_contract.get_gon_participants_rewards().get(self.get_investor_name(), 0.), # GON in strat contract
            'amm_USDT': usdt_amount, # USDT in amm pool
            'amm_GON': gon_amount, # GON in amm pool
            'GON/USDT': amm_pool.reserve1 / amm_pool.reserve0 if amm_pool.reserve0 > 0. else None, # GON/USDT price
            'strat_index_price': strat_contract.get_index_price(), # strat index price
        }
        self._analytics.append(res)

    def mark_to_market(self, datetime, strat_contract, amm_pool):
        '''
        compute several analytics like realised and unrealised pnl on money deployed in both strat_contract and amm_pool
        :param strat_contract: strategy contract object
        :param amm_pool: amm pool contract object
        :return: self
        '''

        strat_price = strat_contract.get_index_price()
        gon_to_usdt_price = amm_pool.reserve1 / amm_pool.reserve0
        lp_to_usdt = (amm_pool.reserve0 * gon_to_usdt_price  + amm_pool.reserve1) / sum(amm_pool.liquidity_providers.values())
        symbol_pairs = self._gon_transactions._pnl_tracker.get_symbol_pairs()
        self._gon_transactions.mark_to_market(symbol_pair='GON/USDT', datetime=datetime, price=gon_to_usdt_price) if\
        'GON/USDT' in symbol_pairs else None
        self._gon_transactions.mark_to_market(symbol_pair='STRATEGY/USDT', datetime=datetime, price=strat_price) if\
        'STRATEGY/USDT' in symbol_pairs else None
        self._gon_transactions.mark_to_market(symbol_pair='STRATEGY/GON', datetime=datetime, price=gon_to_usdt_price) if \
            'STRATEGY/GON' in symbol_pairs else None
        self._gon_transactions.mark_to_market(symbol_pair='AMM/USDT', datetime=datetime, price=lp_to_usdt) if \
            'AMM/USDT' in symbol_pairs else None

        return self

def generate_random_gon_investor(investor_max_cash_usdt, investor_min_cash_usdt, datetime,
                                 initial_cash_usdt=None, name=None,
                                 type=GON_PARTICIPANT_TYPE.COMMUNITY):
    initial_cash_usdt = np.random.randint(investor_min_cash_usdt, investor_max_cash_usdt) \
                        if initial_cash_usdt is None else initial_cash_usdt
    investor = GonParticipant(name=name, type=type)\
               .deposit(amount_usdt=initial_cash_usdt, gas_fee_usdt=0.0, datetime=datetime)
    return investor

def generate_random_gon_investor_deposit_in_strategy(gon_investors_pool, number_of_investors,
                                         strat_contract,
                                         pct_investment_min=.01,
                                         pct_investment_max=.3,
                                         datetime=None):
    gon_investors_retail_strategy_selection = np.random.choice(gon_investors_pool, number_of_investors)
    for retail_strategy_investor in gon_investors_retail_strategy_selection:
        investment_amount_usdt = retail_strategy_investor.invest_in_strategy(
                                                                             pct_of_total_usdt=np.random.randint(pct_investment_min*1e3,
                                                                                                                 pct_investment_max*1e3)/1e3,
                                                                             strat_contract=strat_contract, datetime=datetime)
    return True

