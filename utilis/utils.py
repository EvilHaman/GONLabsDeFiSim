import cvxpy as cp
import numpy as np
import pandas as pd
from amm.amm import ERC20, Factory
import logging
import pandas as pd

B,R,G,O,BL,MAG,CY,WH = range(8)

MY_COLORS = {'ERROR': R, 'DEBUG': BL, 'INFO': G, 'WARNING': O, 'CRITICAL': O}
CS = "\33[%dm"
RS = "\33[0m"

class ColorMyLogger(logging.Formatter):
    def format(self, record) -> str:
        ln = record.levelname
        res = logging.Formatter.format(self, record)
        if ln in MY_COLORS:
            res = CS%(30 + MY_COLORS[ln]) + res + RS
        return res

def setupLogger(log_name, cpu_name=''):
    logi = logging.getLogger(log_name + cpu_name)

    logi.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatting = ColorMyLogger('%(asctime)s : [%(process)d] - %(name)s - %(funcName)s() - %(levelname)s - %(message)s')
    handler.setFormatter(formatting)
    logi.addHandler(handler)
    logi.propagate=False
    return logi

def convert_timestamp_ms_to_datetime(timestamp_ms, floor_to_nearest_minute=True):
    return pd.to_datetime(timestamp_ms, unit='ms', utc=True) if not floor_to_nearest_minute\
           else pd.to_datetime(int((timestamp_ms/1000) // 60 * 60) * 1000, unit='ms', utc=True)

def convert_datetime_to_timstamp_ms(datetime_variable, floor_to_nearest_minute=True):
    res = datetime_variable.timestamp() * 1000
    return int(res) if not floor_to_nearest_minute else int((res/1000) // 60 * 60) * 1000


def monthly_to_daily_returns(monthly_return, trading_days):
    daily_rtn_avg = ((1 + monthly_return) ** (1 / trading_days)) - 1
    daily_rtn_avg_tolerance = 5*1e-4
    # Define variables
    daily_returns = cp.Variable(trading_days)
    obj = cp.Minimize(cp.norm( (cp.sum(daily_returns) / trading_days) - daily_rtn_avg, 2) )
    #obj = cp.Minimize(cp.norm( cp.power(( 1 + (cp.sum(daily_returns) / trading_days) ),trading_days) - 1 - monthly_return, 2))

    # Define constraints
    constraints = []#[cp.sum(daily_returns) == 0]
    for i in range(trading_days):
        constraints.append(daily_returns[i] >= -.1)
        constraints.append(daily_returns[i] <= .1)

    for i in range(trading_days):
        for j in range(i + 1, trading_days):
            constraints.append(daily_returns[i] - daily_returns[j] >= np.random.uniform(-20,20)*1e-4)
            # constraints.append(daily_returns[i] - daily_returns[j] <= - daily_rtn_avg_tolerance)

    constraints.append(cp.sum(daily_returns) / trading_days <= daily_rtn_avg + daily_rtn_avg_tolerance)
    constraints.append(cp.sum(daily_returns) / trading_days >= daily_rtn_avg - daily_rtn_avg_tolerance)

    # Solve the problem
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False)

    # Extract the optimal solution
    daily_returns = np.array(daily_returns.value)

    # Ensure that the solution satisfies the monthly return constraint
    # while ((daily_returns.mean() + 1) ** trading_days - 1) < monthly_return:
    #     max_index = np.argmax(daily_returns)
    #     daily_returns[max_index] = -daily_returns[max_index]

    # Return the solution as a Pandas series
    return pd.Series(daily_returns).sample(frac=1)

def create_amm_pool(coin_name, address, coin_symbol):
    factory = Factory("USDT pool factory", "0x1")
    erc20 = ERC20(coin_name, address)
    usdt = ERC20("usdt", "0x09")
    pool = factory.create_exchange(erc20, usdt, coin_symbol)
    return pool

annualized_rtn = lambda daily_rtn, duration=365: (daily_rtn + 1).prod() ** (1 / (len(daily_rtn) / duration)) - 1

def compute_apy_stats(daily_rtn_series):
    '''
    compute apy stats
    :param daily_rtn_series: pd.Series of daily returns
    :return:
    '''
    return daily_rtn_series.rename('daily_rtn').to_frame().assign(daily_rtn = lambda x: x['daily_rtn'])\
               .assign(apy_7d= lambda x:   x['daily_rtn'].rolling(7)    .apply(lambda y: annualized_rtn(y)),
                       apy_14d= lambda x:  x['daily_rtn'].rolling(14)   .apply(lambda y: annualized_rtn(y)),
                       apy_30d= lambda x:  x['daily_rtn'].rolling(30)   .apply(lambda y: annualized_rtn(y)),
                       apy_60d= lambda x:  x['daily_rtn'].rolling(60)   .apply(lambda y: annualized_rtn(y)),
                       apy_100d= lambda x: x['daily_rtn'].rolling(100, min_periods=60)  .apply(lambda y: annualized_rtn(y)),
                       apy_200d= lambda x: x['daily_rtn'].rolling(200, min_periods=60)  .apply(lambda y: annualized_rtn(y)),
                       apy_365d= lambda x: x['daily_rtn'].rolling(365, min_periods=60)  .apply(lambda y: annualized_rtn(y)),
                        ).rolling(10).mean().fillna(0.)