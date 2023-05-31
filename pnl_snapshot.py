import pandas as pd

class PnlTracker:
    def __init__(self):
        self._symbol_pairs = []
        self._pnl_snapshots = {}

    def get_symbol_pairs(self):
        return self._symbol_pairs.copy()

    def add_symbol_pair(self, symbol_pair):
        self._symbol_pairs.append(symbol_pair)
        self._pnl_snapshots[symbol_pair] = PnlSnapshot(symbol_pair)

        ## NOTE: TEMP MAYBE NEEDED TO AVOID DEVISION BY ZERO ON GON
        self._pnl_snapshots[symbol_pair]._record_analytics(datetime='2019-12-31')
        return self

    def get_pnl_snapshots(self):
        return self._pnl_snapshots.copy()

    def get_pnl_on_symbol_pair(self, symbol_pair):
        if symbol_pair in self.get_symbol_pairs():
            return self._pnl_snapshots[symbol_pair].get_analytics()
        else:
            return None

    def update_by_tradefeed(self, symbol_pair, buy_or_sell, traded_price, traded_quantity, datetime):
        self._pnl_snapshots[symbol_pair].update_by_tradefeed(buy_or_sell, traded_price, traded_quantity, datetime)
        return self

    def update_by_marketdata(self, symbol_pair, last_price, datetime):
        self._pnl_snapshots[symbol_pair].update_by_marketdata(last_price, datetime)
        return self

class PnlSnapshot:
    def __init__(self, ticker):#, buy_or_sell, traded_price, traded_quantity):
        self.m_ticker = ticker
        self.m_net_position = 0
        self.m_avg_open_price = 0.
        self.m_net_investment = 0
        self.m_realized_pnl = 0
        self.m_unrealized_pnl = 0
        self.m_total_pnl = 0
        self._last_price = 0.
        self._last_action = ''
        self._is_still_open = False
        self._analytics_record = []

        #self.update_by_tradefeed(buy_or_sell, traded_price, traded_quantity)

    # buy_or_sell: 1 is buy, 2 is sell
    def update_by_tradefeed(self, buy_or_sell, traded_price, traded_quantity, datetime):
        self._last_action = 'buy' if buy_or_sell == 1 else 'sell'
        # buy: positive position, sell: negative position
        quantity_with_direction = traded_quantity if buy_or_sell == 1 else (-1) * traded_quantity
        self._is_still_open = (self.m_net_position * quantity_with_direction) >= 0

        #print (f'is_still_open: {is_still_open}, (self.m_net_position * quantity_with_direction) = {self.m_net_position * quantity_with_direction}, self.m_net_position = {self.m_net_position}, quantity_with_direction = {quantity_with_direction}')

        # realized pnl
        if not self._is_still_open:
            # Remember to keep the sign as the net position
            self.m_realized_pnl += ( traded_price - self.m_avg_open_price ) *\
                min(
                    abs(quantity_with_direction),
                    abs(self.m_net_position)
                ) * ( abs(self.m_net_position) / self.m_net_position )
        # avg open price
        if self._is_still_open:
            if traded_price == 0:
                #print(f'WARNING: last price is 0, ticker: {self.m_ticker}, datetime: {datetime}')
                self.m_avg_open_price = self.m_avg_open_price * self.m_net_position/( self.m_net_position + quantity_with_direction )
            else:
                self.m_avg_open_price = ( ( self.m_avg_open_price * self.m_net_position ) +
                ( traded_price * quantity_with_direction ) ) / ( self.m_net_position + quantity_with_direction )
        else:
            # Check if it is close-and-open
            if traded_quantity > abs(self.m_net_position):
                self.m_avg_open_price = traded_price
        # net position
        self.m_net_position += quantity_with_direction
        self.m_net_investment = max( self.m_net_investment, abs( self.m_net_position * self.m_avg_open_price  ) )

        # total pnl
        self._last_price = self._last_price if traded_price ==0. else traded_price
        self.m_unrealized_pnl = self.m_net_position * ( self._last_price - self.m_avg_open_price )
        self.m_total_pnl = self.m_realized_pnl + self.m_unrealized_pnl
        self._record_analytics(datetime)
        return self

    def update_by_marketdata(self, last_price, datetime):
        self.m_unrealized_pnl = ( last_price - self.m_avg_open_price ) * self.m_net_position
        self._is_still_open = abs(self.m_net_position) >= 1e-9
        self.m_total_pnl = self.m_realized_pnl + self.m_unrealized_pnl
        self._last_price = last_price
        self._last_action = 'mark-to-market'
        self._record_analytics(datetime)
        return self
    def _record_analytics(self, datetime):
        self._analytics_record.append({
            'datetime': datetime,
            'ticker': self.m_ticker,
            'net_position': self.m_net_position,
            'is_still_open': self._is_still_open,
            'avg_open_price': self.m_avg_open_price,
            'last_price': self._last_price, # 'last_price': self.m_last_price,
            'net_investment': self.m_net_investment,
            'realized_pnl': self.m_realized_pnl,
            'unrealized_pnl': self.m_unrealized_pnl,
            'total_pnl': self.m_total_pnl,
            'last_action': self._last_action
        })
        return self

    def get_analytics(self):
        return pd.DataFrame(self._analytics_record)