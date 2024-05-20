import pandas as pd
import numpy as np
import matplotlib

from utilis.constants import GON_PM_TYPE

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utilis.utils import setupLogger
logger = setupLogger(__name__)

def compute_pnl_attributions(gon_investors_list):
    ## plot pnl of gon investors in the strategy (not including GON rewardes): ###
    strategy_pnl = []
    # gon_investor = gon_investors_retail_strategy[481]
    for gon_investor in gon_investors_list:
        if 'STRATEGY/USDT' in gon_investor._gon_transactions._pnl_tracker.get_symbol_pairs():
            usdt_pnl_strat = gon_investor._gon_transactions.get_pnl_on_symbol_pair('STRATEGY/USDT') \
                .assign(datetime=lambda x: pd.to_datetime(x['datetime'])) \
                .assign(pnl_pct_strat_no_rewards=lambda x: x['total_pnl'] / x['net_investment']).sort_values(by='datetime') \
                .assign(net_position_strat_dlr=lambda x: x['net_position'] * x['last_price']) \
                .groupby('datetime').last()[['pnl_pct_strat_no_rewards', 'net_investment', 'total_pnl', 'is_still_open',
                                             'net_position_strat_dlr']].rename(columns={'total_pnl': 'pnl_strat_no_rewards',
                                                                                        'net_investment': 'net_investment_strat'})
        else:
            usdt_pnl_strat = pd.DataFrame(columns=['pnl_pct_strat_no_rewards', 'net_investment_strat',
                                                   'pnl_strat_no_rewards', 'is_still_open', 'net_position_strat_dlr'])

        if 'STRATEGY/GON' in gon_investor._gon_transactions._pnl_tracker.get_symbol_pairs():
            gon_pnl_strat = gon_investor._gon_transactions.get_pnl_on_symbol_pair('STRATEGY/GON') \
                .pipe(lambda df: df[df['last_action'] != 'buy']) \
                .assign(datetime=lambda x: pd.to_datetime(x['datetime'])) \
                .sort_values(by='datetime') \
                .groupby('datetime').last()
        else:
            gon_pnl_strat = pd.DataFrame(columns=['total_pnl'])

        if 'AMM/USDT' in gon_investor._gon_transactions._pnl_tracker.get_symbol_pairs():
            pnl_amm = gon_investor._gon_transactions.get_pnl_on_symbol_pair('AMM/USDT') \
                .assign(datetime=lambda x: pd.to_datetime(x['datetime'])) \
                .assign(pnl_pct_amm=lambda x: x['total_pnl'] / x['net_investment']).sort_values(by='datetime') \
                .groupby('datetime').last()[['pnl_pct_amm', 'net_investment', 'total_pnl']].rename(
                columns={'net_investment': 'net_investment_amm',
                         'total_pnl': 'pnl_amm'})
        else:
            pnl_amm = pd.DataFrame(columns=['pnl_pct_amm', 'net_investment_amm', 'pnl_amm'])

        if 'GON/USDT' in gon_investor._gon_transactions._pnl_tracker.get_symbol_pairs():
            pnl_swaps = gon_investor._gon_transactions.get_pnl_on_symbol_pair('GON/USDT') \
                .assign(datetime=lambda x: pd.to_datetime(x['datetime'])) \
                .assign(pnl_pct_swaps=lambda x: x['total_pnl'] / x['net_investment']).sort_values(by='datetime') \
                .groupby('datetime').last()[['pnl_pct_swaps', 'net_investment', 'total_pnl']].rename(
                columns={'net_investment': 'net_investment_swaps',
                         'total_pnl': 'pnl_swaps'})
        else:
            pnl_swaps = pd.DataFrame(columns=['pnl_pct_swaps', 'net_investment_swaps', 'pnl_swaps'])

        investor_strat_pnl = usdt_pnl_strat.merge(gon_pnl_strat['total_pnl'].rename('pnl_strat_rewards').to_frame(),
                                                  how='outer', left_index=True, right_index=True) \
            .merge(pnl_amm, how='outer', left_index=True, right_index=True) \
            .merge(pnl_swaps, how='outer', left_index=True, right_index=True) \
            .assign(pnl_strat_rewards=lambda x: x['pnl_strat_rewards'].ffill()) \
            .assign(pnl_pct_strat_rewards=lambda x: x['pnl_strat_rewards'] / x['net_investment_strat']) \
            .assign(pnl_pct_strat_total=lambda x: (x['pnl_strat_no_rewards'] + x['pnl_strat_rewards']) / x[
            'net_investment_strat']) \
            .assign(pnl_total=lambda x: (
                    x['pnl_strat_no_rewards'].fillna(0.).ffill() + x['pnl_strat_rewards'].fillna(0.).ffill() + x[
                'pnl_amm'].fillna(0.).ffill() + x['pnl_swaps'].fillna(0.).ffill())) \
            .assign(pnl_pct_total=lambda x: x['pnl_total'] / \
                                            (x['net_investment_strat'].fillna(0.).ffill() + x[
                                                'net_investment_amm'].fillna(0.).ffill() + x[
                                                 'net_investment_swaps'].fillna(0.).ffill())) \
            .assign(pnl_strat_rewards_with_swaps=lambda x: x['pnl_strat_rewards'] + x['pnl_swaps']) \
            .assign(
            pnl_strat_rewards_with_swaps_pct=lambda x: x['pnl_strat_rewards_with_swaps'] / x['net_investment_strat'])

        pm_type = gon_investor._pm_type
        strategy_pnl.append(investor_strat_pnl.assign(name=gon_investor.get_investor_name(), pm_type=pm_type))

    return strategy_pnl

def principle_stats_on_pm_type(strategy_pnl_on_pm):
    def produce_cumsum_with_reset_on_nan(col):
        cumsum = col.cumsum().fillna(method='pad')
        reset = -cumsum[col.isnull()].diff().fillna(cumsum)
        result = col.where(col.notnull(), reset).cumsum()
        return result

    pnl_pct_strat_total = pd.concat(
        [investor_pnl['pnl_pct_strat_total'].rename(investor_pnl.iloc[0].loc['name']) for investor_pnl in
         strategy_pnl_on_pm], axis=1)
    pnl_strat_total = pd.concat([(investor_pnl['pnl_strat_rewards'] + investor_pnl['pnl_strat_no_rewards']).rename(
        investor_pnl.iloc[0].loc['name']) for investor_pnl in strategy_pnl_on_pm], axis=1)
    net_position_strat_dlr = pd.concat(
        [investor_pnl['net_position_strat_dlr'].rename(investor_pnl.iloc[0].loc['name']) for investor_pnl in
         strategy_pnl_on_pm], axis=1)
    net_invested_strat = pd.concat(
        [investor_pnl['net_investment_strat'].rename(investor_pnl.iloc[0].loc['name']) for investor_pnl in
         strategy_pnl_on_pm], axis=1)
    holding_count = net_position_strat_dlr.where(abs(net_position_strat_dlr) > 1e-9, np.nan).notnull().apply(
        lambda col: produce_cumsum_with_reset_on_nan(col)) \
        .stack().rename('holding_count').to_frame()

    capital_commitment_data = net_position_strat_dlr.where(abs(net_position_strat_dlr) > 1e-9,
                                                           np.nan).stack().rename(
        'net_position_strat_dlr').to_frame() \
        .merge(pnl_pct_strat_total.stack().rename('pnl_pct_strat_total').replace(0., np.nan).to_frame(), how='left',
               left_index=True, right_index=True) \
        .merge(pnl_strat_total.stack().rename('pnl_strat_total').replace(0., np.nan).to_frame(), how='left',
               left_index=True, right_index=True) \
        .merge(net_invested_strat.stack().rename('net_invested_strat').replace(0., np.nan).to_frame(), how='left',
               left_index=True, right_index=True) \
        .merge(holding_count, how='left', left_index=True, right_index=True) \
        .assign(investment_buk=lambda x: pd.cut(x['net_position_strat_dlr'], 20, labels=False)) \
        .assign(holding_buk=lambda x: pd.cut(x['holding_count'], 20, labels=False))

    net_invested_corr = capital_commitment_data.astype(float).corr()['net_invested_strat']
    net_invested_p_values = calculate_pvalues(capital_commitment_data.astype(float))['net_invested_strat']
    holding_count_corr = capital_commitment_data.astype(float).corr()['holding_count']
    holding_count_p_values = calculate_pvalues(capital_commitment_data.astype(float))['holding_count']
    return pd.Series({'net_investment': net_invested_corr.loc['pnl_strat_total'],
                      'holding_count' : holding_count_corr.loc['pnl_strat_total'],
                      'net_investment_p_value': net_invested_p_values.loc['pnl_strat_total'],
                      'holding_count_p_value': holding_count_p_values.loc['pnl_strat_total']})

def calculate_pvalues(df):
    from scipy.stats import pearsonr
    df = df.dropna()._get_numeric_data()
    pvals = pd.DataFrame(index=df.columns, columns=df.columns)
    for r in df.columns:
        for c in df.columns:
            if r == c:
                pvals[r, c] = np.nan
            else:
                _, p = pearsonr(df[r], df[c])
                pvals.at[r, c] = p
    return pvals

def plot_principle_stats(strategy_pnl):
    principle_stats = []
    for pm_type in list(GON_PM_TYPE.to_dict().keys()) + [
        'ALL']:  # [GON_PM_TYPE.SIMPLE, GON_PM_TYPE.SIMPLE_WITH_UNVEST_ON_APY, GON_PM_TYPE.CONSERVATIVE, GON_PM_TYPE.BUY_AND_HOLD]: #'ALL'
        print(pm_type)
        if pm_type != GON_PM_TYPE.LP_PROVIDER:
            if pm_type == 'ALL':
                prince_stats = principle_stats_on_pm_type(strategy_pnl_on_pm=strategy_pnl).rename(pm_type)
            else:
                strategy_pnl_on_pm = [investor_pnl for investor_pnl in strategy_pnl if
                                      investor_pnl.iloc[0].loc['pm_type'] == pm_type]
                prince_stats = principle_stats_on_pm_type(strategy_pnl_on_pm=strategy_pnl_on_pm).rename(pm_type)
            principle_stats.append(prince_stats)

    print('principle_stats: \n', pd.concat(principle_stats, axis=1))
    ax_x = pd.concat(principle_stats, axis=1).iloc[0:2].pipe(lambda df: df * 1e2) \
            .drop(columns=['ALL']).assign(ALL = lambda x: x.mean(axis=1)) \
        .plot(kind='bar', figsize=(15, 10), title='Principle Stats across PM Types', alpha=.5,
              rot=0, fontsize=12, legend=True,
              color=['#BBD9F7', '#F1BA7E', '#EDA8A9', '#EA9A9C', '#C0A5BA', '#33FFD7'])
    ax_x.grid(alpha=.2)
    ax_x.legend()
    ax_x.set_xlabel('PMs', labelpad=15)
    ax_x.set_ylabel('% Correlation with Strategy PnL', labelpad=15)
    plt.show()
########### DEBUG: ###########
# gon_investor._gon_transactions.get_pnl_on_symbol_pair('STRATEGY/USDT') \
#     .assign(datetime=lambda x: pd.to_datetime(x['datetime'])) \
#     .assign(pnl_pct_strat_no_rewards=lambda x: x['total_pnl'] / x['net_investment']).sort_values(by='datetime') \
#     .groupby('datetime').last()\
# .assign(try_this = lambda x: x['last_price']/ x['avg_open_price'] - 1).tail(60)
# for i in range(len(gon_investors_retail_strategy)):
#     try_this = gon_investors_retail_strategy[i]._gon_transactions.get_pnl_on_symbol_pair('STRATEGY/USDT').pipe(lambda df: df[df['last_action'] == 'sell'])
#     if len(try_this) > 0:
#         print (i)
# s = gon_investor._gon_transactions.get_pnl_on_symbol_pair('STRATEGY/USDT').pipe(lambda df: df[df['last_action'] != 'mark-to-market'])\
# .assign(datetime=lambda x: pd.to_datetime(x['datetime']))\
# .set_index('datetime').groupby(level=0).last()\
# .reset_index().sort_values(by='datetime')\
# .set_index(['datetime', 'last_action'])['is_still_open'].unstack()['total_pnl'].fillna(0).cumsum().reset_index()\
# .assign(avg_holding_period = lambda x: x['is_still_open'].replace(False, np.nan)*1)\
# ['avg_holding_period']
# mask = (s == 1)
# s[mask].groupby((~mask).cumsum()).size().tolist()
#
#     .pipe(lambda df: df[df['last_action'] == 'sell'])
#
# data = [1, 1, 1, 1, np.nan, 1, 1, 1, 1, 1, 1, 1, 1, np.nan, np.nan, np.nan, 1, 1, np.nan]
#
# # Create a pandas Series from the Python list
# series = pd.Series(data)
#
# # Create a mask to identify consecutive sequences of 1s
# mask = (series == 1) #& (series.shift(1) == 1)
#
# # Group consecutive sequences and count the number of 1s in each group
# result = series[mask].groupby((~mask).cumsum()).size().tolist()
#
# print(result)

########### DEBUG: ###########

def plot_pnl(pnl_pct, label, color, ax=None, x_label=r'Time (Daily)', y_label=r'% PnL', rolling_days=None, pct_factor=1e2):
    if rolling_days is None:
        pnl_pct_mean = pnl_pct.mean(axis=1) * pct_factor
        pnl_pct_std = pnl_pct.std(axis=1) * pct_factor
        ax = pnl_pct_mean.plot( color = color, label = label, ax=ax)
        ax.fill_between(pnl_pct_mean.index, pnl_pct_mean + pnl_pct_std, pnl_pct_mean - pnl_pct_std , alpha = .1, color = color)
    else:
        pnl_pct_mean = pnl_pct.rolling(rolling_days, center=True).mean() * pct_factor
        pnl_pct_std = pnl_pct.rolling(rolling_days, center=True).std() * pct_factor
        ax = (1e2*pnl_pct).plot(color=color, label=label, ax=ax)
        ax.fill_between(pnl_pct_mean.index, pnl_pct_mean + pnl_pct_std, pnl_pct_mean - pnl_pct_std, alpha=.1,
                        color=color)

    ax.grid(alpha=.2)
    ax.legend()
    ax.set_xlabel(x_label, labelpad=15)
    ax.set_ylabel(y_label, labelpad=15)
    return ax

def plot_pnl_total_per_pm_group(strategy_pnl, gon_pm_types):
    fig, ax = plt.subplots(figsize=(16, 8))
    i = 0
    colors = ['#BBD9F7', '#F1BA7E', '#EDA8A9', '#EA9A9C', '#C0A5BA', 'grey', 'black']
    for pm_type in gon_pm_types:
        strategy_pnl_on_pm = [investor_pnl for investor_pnl in strategy_pnl if
                              investor_pnl.iloc[0].loc['pm_type'] == pm_type]
        pnl_col = pd.concat([investor_pnl['pnl_pct_total'] for investor_pnl in strategy_pnl_on_pm],
                            axis=1).ffill()
        plot_pnl(pnl_pct=pnl_col, label=pm_type, color=colors[i],
                 ax=ax, x_label=r'Time (Daily)', y_label=r'% PnL', rolling_days=None, pct_factor=1e2)
        i += 1

    pnl_col = pd.concat([investor_pnl['pnl_pct_total'] for investor_pnl in strategy_pnl],
                        axis=1).ffill()
    plot_pnl(pnl_pct=pnl_col, label='ALL', color=colors[i],
             ax=ax, x_label=r'Time (Daily)', y_label=r'% PnL', rolling_days=None, pct_factor=1e2)
    plt.title('Gon Participant Pnl Analytics Over time - Per PM Group')
    plt.show()

def compute_stats(rtn_daily, pnl_dlr, pnl_pct):
    res = { 'Cumulative returns' : pnl_pct.iloc[-1],
            'Annual return': (rtn_daily + 1).prod() ** (1 / (len(rtn_daily) / 365)) - 1,
            'Annual volatility' : np.sqrt(365) * rtn_daily.std(),
            'Sharpe' : rtn_daily.agg(lambda x: (x.mean() / x.std() ) * np.sqrt(365) ),
            'Max drawdown': -1 * (1 - (pnl_pct + 1).div(
                (pnl_pct + 1).cummax())).expanding().max().max(),
            }
    return pd.Series(res)

def compute_stats_per_pm_group(strategy_pnl, gon_pm_types):
    res = []
    for pm_type in list(gon_pm_types) + ['ALL']:
        if pm_type == 'ALL':
            strategy_pnl_on_pm = strategy_pnl
        else:
            strategy_pnl_on_pm = [investor_pnl for investor_pnl in strategy_pnl if
                                  investor_pnl.iloc[0].loc['pm_type'] == pm_type]
        pnl_pct = pd.concat([investor_pnl['pnl_pct_total'] for investor_pnl in strategy_pnl_on_pm],
                            axis=1).ffill().mean(axis=1).fillna(0.)

        pnl_dlr =pd.concat([investor_pnl['pnl_total'] for investor_pnl in strategy_pnl_on_pm],
                            axis=1).ffill().mean(axis=1).fillna(0.)

        daily_rtn =  (np.exp(np.log(pnl_pct + 1.0).diff()) - 1.0)

        res.append(compute_stats(rtn_daily=daily_rtn, pnl_dlr=pnl_dlr, pnl_pct=pnl_pct).rename(pm_type))
    return pd.concat(res, axis=1)

def plot_pnl_score_card(strategy_pnl, strategy_roi, gon_to_usdt_eod, title='Gon Participant Pnl Analytics Over time'):
    pnl_plot_setting = pd.DataFrame([
        {'col_name': 'pnl_pct_strat_total', 'color': '#f6a08c', 'label': 'STRAT: TOTAL PNL'},
        {'col_name': 'pnl_pct_strat_no_rewards', 'color': 'darkorchid', 'label': 'STRAT: PNL_ON_USDT'},
        {'col_name': 'pnl_strat_rewards_with_swaps_pct', 'color': '#eb3498', 'label': 'STRAT: PNL_ON_GON'},
        {'col_name': 'pnl_pct_amm', 'color': 'blue', 'label': 'AMM: PNL'},
        # {'col_name': 'pnl_pct_swaps', 'color': 'green', 'label': 'SWAPS: PNL'},
        {'col_name': 'pnl_pct_total', 'color': 'pink', 'label': 'TOTAL PNL'},
    ])

    pnl_plot_setting_dlr = pd.DataFrame([
        # {'col_name': 'pnl_strat_total', 'color': '#f6a08c', 'label': 'STRAT: TOTAL PNL'},
        {'col_name': 'pnl_total', 'color': 'pink', 'label': 'TOTAL PNL'},
        {'col_name': 'pnl_strat_no_rewards', 'color': 'darkorchid', 'label': 'STRAT: PNL_ON_USDT'},
        {'col_name': 'pnl_strat_rewards_with_swaps', 'color': 'brown', 'label': 'STRAT: PNL_ON_GON'},
        # {'col_name': 'pnl_strat_rewards', 'color': '#eb3498', 'label': 'STRAT: PNL_ON_GON'},
        {'col_name': 'pnl_amm', 'color': 'blue', 'label': 'AMM: PNL'},
        # {'col_name': 'pnl_swaps', 'color': 'green', 'label': 'SWAPS: PNL'},
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # plot_pnl(pnl_pct=strategy_roi['Strategy HODL (%)'], label='Strategy HODL (%)', color='black', ax=ax1,
    #          rolling_days=20)
    if strategy_roi['avg_cash_out'].notna().any():
        strategy_roi['avg_cash_out'].plot(color='green', label='Avg Cash Out Target(%)', ax=ax1, style='--', alpha=.8)
        ax1.legend()
        plt.show()

    (1e2 * ((1 + gon_to_usdt_eod.pct_change()).cumprod() - 1)).plot(ax=ax1, label='GON/USDT %', color='#3295a8',
                                                                    alpha=.5)
    ax1.legend()

    ax1_scnd = ax1.twinx()
    is_still_open_stats = pd.concat([investor_pnl['is_still_open'] for investor_pnl in strategy_pnl], axis=1) \
        .pipe(lambda df: 1e2* df.sum(axis=1) / df.count(axis=1))
    ax1_scnd = is_still_open_stats.plot(ax=ax1_scnd, secondary_y=True, label='Open Positions (%)', color='grey',
                                        alpha=.5, style='--')
    ax1_scnd.set_ylabel('% of Strat Open Position', labelpad=15)
    ax1_scnd.grid(alpha=.2)
    ax1_scnd.legend()

    for index, pnl_plot_setting_row in pnl_plot_setting.iterrows():
        pnl_col = pd.concat([investor_pnl[pnl_plot_setting_row.loc['col_name']] for investor_pnl in strategy_pnl],
                            axis=1).ffill()
        plot_ax = plot_pnl(pnl_pct=pnl_col, label=pnl_plot_setting_row.loc['label'],
                           color=pnl_plot_setting_row.loc['color'], ax=ax1)

    # fig, ax = plt.subplots(figsize=(15, 10))
    for index, pnl_plot_setting_row in pnl_plot_setting_dlr.iterrows():
        pnl_col = pd.concat([investor_pnl[pnl_plot_setting_row.loc['col_name']] for investor_pnl in strategy_pnl],
                            axis=1).ffill()
        plot_pnl(pnl_pct=pnl_col, label=pnl_plot_setting_row.loc['label'], color=pnl_plot_setting_row.loc['color'],
                 ax=ax2,
                 y_label=r'$ PnL (Daily)', pct_factor=1)
    # Set a title for the entire figure
    plt.suptitle(title, fontsize=14, fontweight='bold')
    # Adjust the spacing between subplots and figure
    plt.subplots_adjust(top=0.9, wspace=0.35)
    plt.show()

def plot_gon_related_stats(gonomics_engine, GON_total_supply, gon_to_usdt_eod):
    gonomics_buy_n_burn_usdt = gonomics_engine._gon_transactions.get_pnl_on_symbol_pair('GON/USDT').assign(
        datetime=lambda x: pd.to_datetime(x['datetime'])) \
        .groupby('datetime').last()['net_position'].rename('GONOMICS: Buy & Burn')

    distribution_schedule_reformatted = gonomics_engine._gon_distribution_scheduler._GON_distribution_schedule.reset_index().assign(
        month=lambda x: x['month'] + 1) \
        .drop(columns=['datetime', 'TOTAL']).pipe(
        lambda df: df.append(pd.Series(df.iloc[0] * 0), ignore_index=True)).set_index('month').sort_index()

    fig, ( (ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, figsize=(20, 15))

    simulated_distribution_stats = (distribution_schedule_reformatted * gonomics_engine._gon_distribution_scheduler._GON_distribution * GON_total_supply) \
        .merge(gon_to_usdt_eod.resample('1m').last().reset_index().drop(columns='datetime')['gon_price'], how='left',
               left_index=True, right_index=True) \
        .merge(gonomics_buy_n_burn_usdt.resample('1m').last().reset_index().drop(columns='datetime'), how='left', left_index=True, right_index=True) \
        .pipe(lambda df: df.apply(lambda col: col * df['gon_price'])).drop(columns='gon_price').ffill() \
        .assign(market_cap=lambda x: x.drop(columns='GONOMICS: Buy & Burn').sum(axis=1))

    simulated_distribution_stats.plot(kind='line', figsize=(15, 10), alpha=.5, title='GON Simulated Distribution, Buy & Burn, and Market-Cap (Over Time)',
              color=['#fc03a9', 'pink', '#BAA16E', '#4EDA87', '#FF5733', '#FFD533', '#33FFD7', '#33B9FF'], ax=ax11)
    ax11.grid(alpha=.2)

    ax11.legend()
    ax11.set_xlabel('Time (Monthly)', labelpad=15)
    ax11.set_ylabel('USDT', labelpad=15)

    simulated_distribution_stats.iloc[-1].plot(kind='bar', figsize=(15, 10), alpha=.5, title='GON Simulated Distribution, Buy & Burn, and Market-Cap (Total)',
                color=['#fc03a9', 'pink', '#BAA16E', '#4EDA87', '#FF5733', '#FFD533', '#33FFD7', '#33B9FF'], ax=ax12)
    ax12.grid(alpha=.2)
    ax12.set_ylabel('USDT', labelpad=15)

    ax13 = gon_to_usdt_eod.rename('GON/USDT').plot(kind='line', figsize=(15, 10), alpha=.5, title='GON/USDT (Over Time)', color='pink', ax=ax13)
    ax13_scnd = ax13.twinx()
    ax13.legend()
    ax13.set_xlabel('Time (Daily)', labelpad=15)
    ax13.set_ylabel('% Rtn', labelpad=15)

    ((1 + gon_to_usdt_eod.pct_change() ).cumprod() - 1).pipe(lambda s: s*1e2).fillna(0.).plot(kind='line', figsize=(15, 10), alpha=.5, color='pink',
                         ax=ax13_scnd, secondary_y=True)
    ax13_scnd.grid(alpha=.2)
    ax13_scnd.set_ylabel('% Rtn (Cumulative)', labelpad=15)
    plt.show()

def plot_gonomics_distribution_plan(gonomics_engine, GON_total_supply, gon_to_usdt_eod, community_map):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 15))
    distribution_schedule_reformatted = gonomics_engine._gon_distribution_scheduler._GON_distribution_schedule.reset_index().assign(
        month=lambda x: x['month'] + 1) \
        .drop(columns=['datetime', 'TOTAL']).pipe(
        lambda df: df.append(pd.Series(df.iloc[0] * 0), ignore_index=True)).set_index('month').sort_index()

    (distribution_schedule_reformatted * gonomics_engine._gon_distribution_scheduler._GON_distribution * GON_total_supply) \
        .plot(kind='area', figsize=(15, 10), alpha=.5,
              color=['#fc03a9', 'pink', '#BAA16E', '#4EDA87', '#FF5733', '#FFD533', '#33FFD7', '#33B9FF'], ax=ax1)
    ax1.grid(alpha=.2)

    ax1.legend()
    ax1.set_xlabel('# Month', labelpad=15)
    ax1.set_ylabel('# GON', labelpad=15)

    (distribution_schedule_reformatted * gonomics_engine._gon_distribution_scheduler._GON_distribution * 1e2) \
        .plot(kind='area', color=['#fc03a9', 'pink', '#BAA16E', '#4EDA87', '#FF5733', '#FFD533', '#33FFD7', '#33B9FF'],
              alpha=.5, figsize=(15, 10), ax=ax2)
    ax2.set_ylabel('% of GON', labelpad=15)
    ax2.grid(alpha=.2)
    ax2.legend()
    #plt.show()

    gonomics_engine._gon_distribution_scheduler._GON_distribution \
        .plot(autopct="%.1f%%", figsize=(15, 10), fontsize=10, kind='pie', legend=False, labeldistance=1.1,
              pctdistance=0.8,
              colors=['#fc03a9', 'pink', '#BAA16E', '#4EDA87', '#FF5733', '#FFD533', '#33FFD7', '#33B9FF'],
              wedgeprops={"alpha": 0.5}, ax=ax3)

    plt.show()

    fig, ax = plt.subplots(figsize=(16, 8))
    community_map.plot(autopct="%.1f%%", figsize=(15, 10), fontsize=10, kind='pie', legend=False, labeldistance=1.1,
          pctdistance=0.8,
          colors=['#BBD9F7', '#F1BA7E', '#EDA8A9', '#EA9A9C', '#C0A5BA'],
          wedgeprops={"alpha": 0.5}, ax=ax)
    plt.show()
