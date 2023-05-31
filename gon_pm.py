import pandas as pd
import warnings

from utilis.constants import GON_PARTICIPANT_TYPE, GON_PM_TYPE
from gon_participant import GonParticipant

warnings.simplefilter(action='ignore', category=FutureWarning)
from utilis.utils import setupLogger
import numpy as np
logger = setupLogger(__name__)
class GonPortfolioManagerBase(GonParticipant):
    '''
    this is a class designed to create sensible strategies of gon participants in the defi ecosystem.
    the base class will take a gon participant, and decide on his next actions.
    then the gon participant will execute the actions.
    '''
    def __init__(self, name=None, type=GON_PARTICIPANT_TYPE.COMMUNITY,
                 GAS_FEE_USDT = 15.0, pm_type=GON_PM_TYPE.SIMPLE):
        self._pm_type = pm_type
        super(GonPortfolioManagerBase, self).__init__(name=name, type=type, GAS_FEE_USDT=GAS_FEE_USDT)

    def set_local_params(self, **kwargs):
        '''
        this function will set the local params of the strategy
        :param kwargs: dict of params
        :return: self
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def on_start(self, strat_contract, amm_pool, datetime):
        '''
        this function will be called by the gon participant on start
        :param strat_contract: strategy contract
        :param amm_pool: amm pool
        :param datetime: datetime of the market data
        :return: self
        '''
        raise NotImplementedError('this function should be implemented by the child class')

    def what_todo_today(self, strat_contract, amm_pool, datetime, **kwargs):
        '''
        this function will be called by the gon participant to get the next action to execute
        :param market_data: dict of symbol_pair: price
        :param datetime: datetime of the market data
        :return: dict of symbol_pair: action
        '''
        raise NotImplementedError('this function should be implemented by the child class')

class GonPortfolioManagerSimple(GonPortfolioManagerBase):
    '''
    this is a simple strategy that will deposit cash every 2-3 weeks, at the order of 200-400 USDT, and sell all when double the money
    '''
    def __init__(self, name=None, type=GON_PARTICIPANT_TYPE.COMMUNITY, GAS_FEE_USDT = 15.0, pm_type=GON_PM_TYPE.SIMPLE):
        super(GonPortfolioManagerSimple, self).__init__(name=name, type=type, GAS_FEE_USDT=GAS_FEE_USDT, pm_type=pm_type)

    def set_local_params(self, days_since_last_trade=0., strat_buy_yield_pct=100.):
        kwargs = {'_days_since_last_trade': days_since_last_trade, '_strat_buy_yield_pct': strat_buy_yield_pct}
        super(GonPortfolioManagerSimple, self).set_local_params(**kwargs)
        return self

    def on_start(self, strat_contract, amm_pool, datetime):
        total_deposit = np.random.randint(200, 400)
        self.deposit(amount_usdt=total_deposit, datetime=datetime)
        self.invest_in_strategy(pct_of_total_usdt=total_deposit / self._USDT,
                                strat_contract=strat_contract,
                                datetime=datetime)

    def what_todo_today(self, strat_contract, amm_pool, datetime, **kwargs):
        strat_usdt_pnl = self._gon_transactions.get_pnl_on_symbol_pair('STRATEGY/USDT')
        strat_unrealised_pnl = strat_usdt_pnl['unrealized_pnl'].iloc[-1]
        strat_position_usdt = strat_usdt_pnl['net_position'].iloc[-1]
        avg_open_price = strat_usdt_pnl['avg_open_price'].iloc[-1]
        strat_pct_of_total_usdt = strat_unrealised_pnl/ (strat_position_usdt*avg_open_price) if abs(strat_position_usdt) > 1e-9 else 0

        if strat_pct_of_total_usdt >= self._strat_buy_yield_pct/1e2:
            logger.debug(f'{self.get_investor_name()} is selling all his position in the strategy, strat_pct_of_total_usdt={strat_pct_of_total_usdt},'
                         f' strat_buy_yield_pct={self._strat_buy_yield_pct}, strat_position_usdt={strat_position_usdt}, strat_unrealised_pnl={strat_unrealised_pnl}')
            self.unvest_in_strategy(strat_contract=strat_contract,
                                            gon_to_usdt=amm_pool.reserve1 / amm_pool.reserve0,
                                            datetime=datetime)
            self.swap(pct_of_total=1., amm_pool=amm_pool, datetime=datetime, account='GON')
            self._days_since_last_trade = 0

        elif self._days_since_last_trade > np.random.randint(15, 28):
            self._days_since_last_trade = 0
            total_deposit = np.random.randint(200, 400)
            logger.debug(
                f'{self.get_investor_name()} is depositing cash to the strategy, days_since_last_trade={self._days_since_last_trade}'
                f' total_deposit={total_deposit}')
            self.deposit(amount_usdt=total_deposit, datetime=datetime)
            self.invest_in_strategy(pct_of_total_usdt=total_deposit/self._USDT,
                                                     strat_contract=strat_contract,
                                                     datetime=datetime)
        else:
            self._days_since_last_trade += 1
        return self

class GonPortfolioManagerDoNothing(GonPortfolioManagerBase):
    '''
    this is a simple strategy that will deposit cash every 2-3 weeks, at the order of 200-400 USDT, and sell all when double the money
    '''
    def on_start(self, strat_contract, amm_pool, datetime):
        pass

    def what_todo_today(self, strat_contract, amm_pool, datetime, **kwargs):
        pass

class GonPortfolioManagerLiliquidityProvider(GonPortfolioManagerBase):
    '''
    this is a simple strategy that will deposit cash every 2-3 weeks, at the order of 200-400 USDT, and sell all when double the money
    '''

    def set_local_params(self, initial_amm_tvl, target_3d_pct):
        kwargs = {'_initial_amm_tvl': initial_amm_tvl, '_days_since_last_trade': 0., '_target_3d_pct': target_3d_pct}
        super(GonPortfolioManagerLiliquidityProvider, self).set_local_params(**kwargs)
        return self

    def on_start(self, strat_contract, amm_pool, datetime):
        pass

    def what_todo_today(self, strat_contract, amm_pool, datetime, **kwargs):
        gon_to_usdt_tracker = kwargs.get('gon_to_usdt_tracker', None)
        if gon_to_usdt_tracker is None:
            raise ValueError('gon_to_usdt_tracker should be provided in kwargs')
        gon_to_usdt_rtn_3d = ((1 + gon_to_usdt_tracker.tail(3).pct_change()).cumprod() - 1).iloc[-1]
        if gon_to_usdt_rtn_3d < self._target_3d_pct and self._days_since_last_trade > 3:
            logger.warning(f'{self.get_investor_name()} entering to support GON price since, gon_to_usdt_rtn_3d={gon_to_usdt_rtn_3d} and smaller than 5%')
            self.swap(pct_of_total=0.1, amm_pool=amm_pool, datetime=datetime, account='USDT')
            self._days_since_last_trade = 0
        else:
            self._days_since_last_trade += 1
        return self

class GonPortfolioManagerBuyAndHold(GonPortfolioManagerBase):
    '''
    this is a simple strategy that will deposit cash every 2-3 weeks, at the order of 200-400 USDT, and sell all when double the money
    '''
    def on_start(self, strat_contract, amm_pool, datetime):
        total_deposit = np.random.randint(500, 1000)
        self.deposit(amount_usdt=total_deposit, datetime=datetime)
        self.invest_in_strategy(pct_of_total_usdt=total_deposit / self._USDT,
                                strat_contract=strat_contract,
                                datetime=datetime)

    def what_todo_today(self, strat_contract, amm_pool, datetime, **kwargs):
        pass

class GonPortfolioManagerSimpleWithProfitTakingOnApy(GonPortfolioManagerSimple):
    def what_todo_today(self, strat_contract, amm_pool, datetime, **kwargs):
        start_apy_stats = kwargs.get('start_apy_stats', None)
        if start_apy_stats is None:
            raise ValueError('start_apy_stats should be provided in kwargs')
        strat_usdt_pnl = self._gon_transactions.get_pnl_on_symbol_pair('STRATEGY/USDT')
        strat_unrealised_pnl = strat_usdt_pnl['unrealized_pnl'].iloc[-1]
        strat_position_usdt = strat_usdt_pnl['net_position'].iloc[-1]
        avg_open_price = strat_usdt_pnl['avg_open_price'].iloc[-1]
        strat_pct_of_total_usdt = strat_unrealised_pnl/ (strat_position_usdt*avg_open_price) if abs(strat_position_usdt) > 1e-9 else 0

        if strat_pct_of_total_usdt >= self._strat_buy_yield_pct/1e2 or  start_apy_stats.loc['apy_60d'] < 0.:
            logger.debug(f'{self.get_investor_name()} is selling all his position in the strategy, strat_pct_of_total_usdt={strat_pct_of_total_usdt},'
                         f' strat_buy_yield_pct={self._strat_buy_yield_pct}, strat_position_usdt={strat_position_usdt}, strat_unrealised_pnl={strat_unrealised_pnl}')
            self.unvest_in_strategy(strat_contract=strat_contract,
                                            gon_to_usdt=amm_pool.reserve1 / amm_pool.reserve0,
                                            datetime=datetime)
            self.swap(pct_of_total=1., amm_pool=amm_pool, datetime=datetime, account='GON')
            self._days_since_last_trade = 0

        elif self._days_since_last_trade > np.random.randint(15, 28) and start_apy_stats.loc['apy_60d'] > 0.:
            self._days_since_last_trade = 0
            total_deposit = np.random.randint(200, 400)
            logger.debug(
                f'{self.get_investor_name()} is depositing cash to the strategy, days_since_last_trade={self._days_since_last_trade}'
                f' total_deposit={total_deposit}')
            self.deposit(amount_usdt=total_deposit, datetime=datetime)
            self.invest_in_strategy(pct_of_total_usdt=total_deposit/self._USDT,
                                                     strat_contract=strat_contract,
                                                     datetime=datetime)
        else:
            self._days_since_last_trade += 1
        return self

class GonPortfolioManagerSimpleConservative(GonPortfolioManagerSimple):
    def set_local_params(self):
        kwargs = {'_days_since_last_trade': 0}
        super(GonPortfolioManagerSimple, self).set_local_params(**kwargs)
        return self

    def on_start(self, strat_contract, amm_pool, datetime):
        total_deposit = 2000
        self.deposit(amount_usdt=total_deposit, datetime=datetime)
        self.invest_in_strategy(pct_of_total_usdt=total_deposit / self._USDT,
                                strat_contract=strat_contract,
                                datetime=datetime)

    def what_todo_today(self, strat_contract, amm_pool, datetime, **kwargs):
        start_apy_stats = kwargs.get('start_apy_stats', None)
        if start_apy_stats is None:
            raise ValueError('start_apy_stats should be provided in kwargs')
        strat_usdt_pnl = self._gon_transactions.get_pnl_on_symbol_pair('STRATEGY/USDT')
        strat_unrealised_pnl = strat_usdt_pnl['unrealized_pnl'].iloc[-1]
        strat_position_usdt = strat_usdt_pnl['net_position'].iloc[-1]
        avg_open_price = strat_usdt_pnl['avg_open_price'].iloc[-1]
        strat_pct_of_total_usdt = strat_unrealised_pnl/ (strat_position_usdt*avg_open_price) if abs(strat_position_usdt) > 1e-9 else 0

        apy_7d = start_apy_stats.loc['apy_7d']
        if strat_unrealised_pnl >= 5*self.GAS_FEE_USDT or  apy_7d < 0.:
            logger.debug(f'{self.get_investor_name()} is selling all his position in the strategy, strat_pct_of_total_usdt={strat_pct_of_total_usdt},'
                            f' apy_7d={apy_7d}, strat_position_usdt={strat_position_usdt}, strat_unrealised_pnl={strat_unrealised_pnl}')
            self.unvest_in_strategy(strat_contract=strat_contract,
                                            gon_to_usdt=amm_pool.reserve1 / amm_pool.reserve0,
                                            datetime=datetime)
            self.swap(pct_of_total=1., amm_pool=amm_pool, datetime=datetime, account='GON')
            self._days_since_last_trade = 0

        elif apy_7d > 0.:
            self._days_since_last_trade = 0
            total_deposit = np.random.randint(200, 400)
            logger.debug(
                f'{self.get_investor_name()} is depositing cash to the strategy, days_since_last_trade={self._days_since_last_trade}'
                f' total_deposit={total_deposit}')
            self.deposit(amount_usdt=total_deposit, datetime=datetime)
            self.invest_in_strategy(pct_of_total_usdt=total_deposit/self._USDT,
                                                     strat_contract=strat_contract,
                                                     datetime=datetime)
        else:
            self._days_since_last_trade += 1
        return self

class GonPortfolioManagerSophisticated(GonPortfolioManagerSimple):
    def on_start(self, strat_contract, amm_pool, datetime):
        total_deposit = 2000
        self.deposit(amount_usdt=total_deposit, datetime=datetime)
        self.invest_in_strategy(pct_of_total_usdt=total_deposit / self._USDT,
                                strat_contract=strat_contract,
                                datetime=datetime)

        self._apy_strat_unvest_pct =  np.random.randint(30, 50)/100
        self._gon_to_usdt_rtn_30d = np.random.randint(30, 50)/100

    def what_todo_today(self, strat_contract, amm_pool, datetime, **kwargs):
        start_apy_stats = kwargs.get('start_apy_stats', None)
        gon_to_usdt_tracker = kwargs.get('gon_to_usdt_tracker', None)
        if start_apy_stats is None or gon_to_usdt_tracker is None:
            raise ValueError('start_apy_stats or gon_to_usdt_tracker should be provided in kwargs')
        gon_to_usdt_rtn_30d = ((1 + gon_to_usdt_tracker.tail(30).pct_change()).cumprod() - 1).iloc[-1]

        strat_usdt_pnl = self._gon_transactions.get_pnl_on_symbol_pair('STRATEGY/USDT')
        strat_unrealised_pnl = strat_usdt_pnl['unrealized_pnl'].iloc[-1]
        strat_position_usdt = strat_usdt_pnl['net_position'].iloc[-1]
        avg_open_price = strat_usdt_pnl['avg_open_price'].iloc[-1]
        strat_pct_of_total_usdt = strat_unrealised_pnl/ (strat_position_usdt*avg_open_price) if abs(strat_position_usdt) > 1e-9 else 0

        # invest if some time has past and the 60d apy of the strategy is good:
        if self._days_since_last_trade > np.random.randint(5, 10) and start_apy_stats.loc['apy_60d'] >= self._apy_strat_unvest_pct:
            total_deposit = np.random.randint(500, 1000)
            logger.debug(
                f'{self.get_investor_name()} is depositing cash to the strategy, days_since_last_trade={self._days_since_last_trade}'
                f' total_deposit={total_deposit}')
            self.deposit(amount_usdt=total_deposit, datetime=datetime)
            self.invest_in_strategy(pct_of_total_usdt=total_deposit / self._USDT,
                                    strat_contract=strat_contract,
                                    datetime=datetime)

            total_deposit = np.random.randint(500, 1000)
            logger.debug(
                f'{self.get_investor_name()} is depositing cash to the strategy, days_since_last_trade={self._days_since_last_trade}'
                f' total_deposit={total_deposit}')
            self.deposit(amount_usdt=total_deposit, datetime=datetime)
            self.add_amm_liquidity(amm_pool=amm_pool, pct_of_total=total_deposit / self._USDT, datetime=datetime)
            self._days_since_last_trade = 0

        elif self._days_since_last_trade > np.random.randint(5, 10) and start_apy_stats.loc['apy_60d'] < self._apy_strat_unvest_pct:
            logger.debug(f'{self.get_investor_name()} is selling all his position in the strategy, strat_pct_of_total_usdt={strat_pct_of_total_usdt},'
                         f' strat_buy_yield_pct={self._strat_buy_yield_pct}, strat_position_usdt={strat_position_usdt}, strat_unrealised_pnl={strat_unrealised_pnl}')
            self.unvest_in_strategy(strat_contract=strat_contract,
                                            gon_to_usdt=amm_pool.reserve1 / amm_pool.reserve0,
                                            datetime=datetime)
            self._days_since_last_trade = 0

        if  self._days_since_last_trade > np.random.randint(5, 10) and gon_to_usdt_rtn_30d < 0.:
            total_deposit = np.random.randint(500, 1000)
            self.deposit(amount_usdt=total_deposit, datetime=datetime)
            self.swap(pct_of_total=total_deposit / self._USDT, amm_pool=amm_pool, datetime=datetime, account='USDT')
            self.add_amm_liquidity(amm_pool=amm_pool, pct_of_total=total_deposit / self._USDT, datetime=datetime)

        elif self._days_since_last_trade > np.random.randint(5, 10) and gon_to_usdt_rtn_30d > self._gon_to_usdt_rtn_30d:
            self.withdraw_amm_liquidity(amm_pool=amm_pool, datetime=datetime)
            self.swap(pct_of_total=.5, amm_pool=amm_pool, datetime=datetime, account='GON')

        self._days_since_last_trade += 1
        return self