import pandas as pd
import warnings

from utilis.constants import GON_PARTICIPANT_ACTION_TYPE
from pnl_snapshot import PnlTracker

warnings.simplefilter(action='ignore', category=FutureWarning)
from utilis.utils import setupLogger
logger = setupLogger(__name__)

class GonTransactions:
    '''
    creating class EvilTrades. the class will store the trades of the evil investor,
    '''
    def __init__(self):
        self._trades = pd.DataFrame(columns=['symbol', 'symbol_pair', 'side', 'amount', 'price', 'datetime'])
        self._pnl_tracker = PnlTracker()

    def getTrades(self):
        '''
        get the trades of the evil investor
        :return: trades: pd.dataframe of the trades of the evil portfolio
        '''
        return self._trades.copy()

    def recordTrade(self, symbol, symbol_pair, side, amount, price, datetime, gas_fee_usdt=0., action_type=GON_PARTICIPANT_ACTION_TYPE.INVEST_IN_STRATEGY):
        trade = pd.Series({'symbol': symbol, 'symbol_pair': symbol_pair,
                                            'side': side, 'amount': amount, 'price': price,
                                            'gas_fee_usdt': gas_fee_usdt,
                                            'datetime': datetime,
                                            'action_type':action_type})
        self._trades = self._trades.append(trade, ignore_index=True)

        ## add to pnl tracker the trade, unless already recorded
        self._pnl_tracker.add_symbol_pair(symbol_pair=symbol_pair) if\
        symbol_pair not in self._pnl_tracker.get_symbol_pairs() else None

        self._pnl_tracker.update_by_tradefeed(symbol_pair=symbol_pair,
                                              buy_or_sell=1 if side == 'BUY' else -1,
                                              traded_price=price,
                                              traded_quantity=amount,
                                                datetime=datetime)
        return self

    def mark_to_market(self, symbol_pair, price, datetime):
        self._pnl_tracker.update_by_marketdata(symbol_pair, price, datetime)
        return self

    def get_pnl_on_symbol_pair(self, symbol_pair):
        return self._pnl_tracker.get_pnl_on_symbol_pair(symbol_pair)