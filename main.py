import pandas as pd
import numpy as np

from sim_func import run_simulation
from utilis.utils import setupLogger
from analytics_and_plots.analytics_aggregate import compute_pnl_attributions, plot_pnl, plot_pnl_score_card, \
    plot_pnl_total_per_pm_group, compute_stats_per_pm_group, plot_gonomics_distribution_plan, \
    plot_principle_stats, plot_gon_related_stats
from utilis.constants import GON_PM_TYPE
from gon_participant import GON_PARTICIPANT_TYPE
from gon_pm import GonPortfolioManagerSimple, GonPortfolioManagerSimpleWithProfitTakingOnApy, \
    GonPortfolioManagerSimpleConservative, GonPortfolioManagerBuyAndHold, GonPortfolioManagerLiliquidityProvider, \
    GonPortfolioManagerSophisticated, GonPortfolioManagerDoNothing
from gonomics import GonomicsEngine
from strategy_contract import GonStrategyContract
from utilis.utils import create_amm_pool
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logger = setupLogger(__name__)

#create a distribution plan from month 1 to 24 months
GON_distribution = pd.Series({GON_PARTICIPANT_TYPE.FOUNDING_TEAM: 0.25,
                    GON_PARTICIPANT_TYPE.PRE_SEEDER: 0.125,
                    GON_PARTICIPANT_TYPE.LP_PROVIDER: .125,
                    GON_PARTICIPANT_TYPE.COMMUNITY: 0.5,})

GON_distribution_schedule = pd.DataFrame(
                             {GON_PARTICIPANT_TYPE.FOUNDING_TEAM:            {0: 0.1, 6: .25, 13: .5,  24:.75, 36:1.0},
                              GON_PARTICIPANT_TYPE.PRE_SEEDER   :            {0: 0.125, 6: .25, 13: .5,  24:.75, 36:1.0},
                              GON_PARTICIPANT_TYPE.LP_PROVIDER:              {0: 0.125, 6: .25, 13: .5,  24:.75, 36:1.0},
                              GON_PARTICIPANT_TYPE.COMMUNITY:                {0: 0.5,         13: .75, 24:.85, 36:1.0}
                             })

#simulator initial conditions and settings:
GON_total_supply = 200*1e6 # 200M GON tokens
initial_amm_tvl = 100*1e3 # 100K USDT

# strategy investors\retails setting:
number_of_investors = 1000 #number of investors in the simulation that will have various strategies of deployment in the system
investor_max_cash = 5000
investor_min_cash = 500

# this is just for fun, not in use yet:
# GON_init_price =  initial_strategy_tvl/GON_total_supply
# market_cap_diluted = GON_total_supply * GON_init_price

## setup special participants:
gon_founders = GonPortfolioManagerDoNothing(name='gon_founders',
                                                            type=GON_PARTICIPANT_TYPE.FOUNDING_TEAM,
                                                         pm_type=GON_PM_TYPE.FOUNDING_TEAM)

gon_pre_seeders =  GonPortfolioManagerDoNothing(name='gon_pre_seeders',
                                                            type=GON_PARTICIPANT_TYPE.PRE_SEEDER,
                                                         pm_type=GON_PM_TYPE.PRE_SEEDER)

# initial amm pool os GON USDT from lp_provider:
gon_lp_provider = GonPortfolioManagerLiliquidityProvider(name='gon_lp_provider',
                                                            type=GON_PARTICIPANT_TYPE.LP_PROVIDER,
                                                         pm_type=GON_PM_TYPE.LP_PROVIDER).set_local_params(initial_amm_tvl=initial_amm_tvl, target_3d_pct=.05)\
                   .deposit(amount_usdt=initial_amm_tvl, gas_fee_usdt=0., datetime='2019-12-31')
amm_pool = create_amm_pool(coin_name='GONLabToken', address='0x111', coin_symbol='GON')
gon_lp_provider._GON += GON_total_supply*GON_distribution_schedule.loc[0][GON_PARTICIPANT_TYPE.LP_PROVIDER]
gon_lp_provider.add_amm_liquidity(pct_of_total=1., amm_pool=amm_pool, datetime='2019-12-31')
## depositing equal amount of USDT to the LP provider, for supporting GON price:
gon_lp_provider.deposit(amount_usdt=initial_amm_tvl, gas_fee_usdt=0., datetime='2019-12-31')

# create a pool of GON PMs (community of PMs):
gon_investors_pct_map = {
    GON_PM_TYPE.SIMPLE:                     { 'pct':.3,
                                              'func': lambda x: GonPortfolioManagerSimple\
                                                                (name=None, type=GON_PARTICIPANT_TYPE.COMMUNITY, pm_type=GON_PM_TYPE.SIMPLE)\
                                                                .set_local_params(strat_buy_yield_pct=np.random.randint(20, 70)) },
    GON_PM_TYPE.SIMPLE_PP: { 'pct':.3,
                                              'func': lambda x: GonPortfolioManagerSimpleWithProfitTakingOnApy\
                                                                (name=None, type=GON_PARTICIPANT_TYPE.COMMUNITY, pm_type=GON_PM_TYPE.SIMPLE_PP)\
                                                                .set_local_params(strat_buy_yield_pct=np.random.randint(20, 35)) },
    GON_PM_TYPE.CONSERVATIVE:               { 'pct':.15,
                                              'func': lambda x: GonPortfolioManagerSimpleConservative\
                                                                (name=None, type=GON_PARTICIPANT_TYPE.COMMUNITY, pm_type=GON_PM_TYPE.CONSERVATIVE)\
                                                                .set_local_params() },
    GON_PM_TYPE.BUY_AND_HOLD:               {'pct': .05,
                                             'func': lambda x: GonPortfolioManagerBuyAndHold\
                                                                (name=None, type=GON_PARTICIPANT_TYPE.COMMUNITY, pm_type=GON_PM_TYPE.BUY_AND_HOLD) \
                                                                .set_local_params()},
    GON_PM_TYPE.SOPHISTICATED:              {'pct': .2,
                                             'func': lambda x: GonPortfolioManagerSophisticated\
                                                                (name=None, type=GON_PARTICIPANT_TYPE.COMMUNITY, pm_type=GON_PM_TYPE.SOPHISTICATED) \
                                                                .set_local_params()},
}
assert abs(sum( gon_investors_pct_map[i]['pct'] for i in gon_investors_pct_map.keys()) - 1.0) < 1e-10, 'investors pct map must sum to 1.0'
gon_investors_retail_strategy = []
for pm_type, pct in gon_investors_pct_map.items():
    number_of_pms = int(gon_investors_pct_map[pm_type]['pct']*number_of_investors)
    gon_investors_retail_strategy += [ gon_investors_pct_map[pm_type]['func'](None)  for i in range(number_of_pms)]
#gon_investors_retail_strategy = [ GonPortfolioManagerSimple(name=None, type=GON_PARTICIPANT_TYPE.COMMUNITY).set_local_params(strat_buy_yield_pct=np.random.randint(20, 70)) for i in range(number_of_investors)]

# pick a strategy performance to be used in the simulation:
rtn_daily = pd.read_pickle('resources\\rtn_daily-Crypto-Systematic-Fundamental.pkl')

# setup strategy contract and deposit 100% of the strategy capital:
reward_fee_on_pnl_bps = 200
strat_contract = GonStrategyContract(strategy_name = 'Crypto-Systematic-Fundamental',
                                     reward_fee_on_pnl_bps=reward_fee_on_pnl_bps)

## set the Gonomics engine:
gonomics_engine = GonomicsEngine(gon_total_supply=GON_total_supply, gas_fee_usdt=15)\
.set_distribution_schedule(GON_distribution=GON_distribution, GON_distribution_schedule=GON_distribution_schedule,
                           datetime_first=pd.to_datetime(rtn_daily.index[0]))


######################################### SIM RUN ##################################################################
res = run_simulation(rtn_daily, strat_contract, gonomics_engine, amm_pool, gon_investors_retail_strategy,
                   gon_lp_provider, gon_pre_seeders, gon_founders, sim_start_date='2019-12-31')

################################## results ########################################################################
strategy_pnl = compute_pnl_attributions(gon_investors_list=[gon_lp_provider] + gon_investors_retail_strategy)

################ loyalty and capital checks ################
#TODO: use this to compute principle stats:
plot_principle_stats(strategy_pnl)

################ Plot PnL per PM: ################
gon_to_usdt_eod = \
    pd.concat(res, axis=1).T.assign(datetime=lambda x: pd.to_datetime(x['today'])).set_index('datetime')['gon_price']

strategy_roi = pd.concat(res, axis=1).T \
    .assign(datetime=lambda x: pd.to_datetime(x['today'])) \
    .set_index('datetime') \
    .assign(strategy_roi=lambda x: (1 + x['daily_rtn']).cumprod() - 1)['strategy_roi'].rename('Strategy HODL (%)') \
    .to_frame()#

######################## use the below to generate analytics Score Card per PM group:#######################
# for pm_type in list(GON_PM_TYPE.to_dict().keys()) + ['ALL']:#[GON_PM_TYPE.SIMPLE, GON_PM_TYPE.SIMPLE_WITH_UNVEST_ON_APY, GON_PM_TYPE.CONSERVATIVE, GON_PM_TYPE.BUY_AND_HOLD]: #'ALL'
#     print (pm_type)
#     if pm_type == 'ALL':
#         plot_pnl_score_card(strategy_pnl=strategy_pnl,
#                             strategy_roi=strategy_roi.assign(avg_cash_out=None),
#                             gon_to_usdt_eod=gon_to_usdt_eod,
#                             title=f'Gon Participant Pnl Analytics Over time - {pm_type}')
#     else:
#         strategy_pnl_on_pm = [investor_pnl for investor_pnl in strategy_pnl if investor_pnl.iloc[0].loc['pm_type'] == pm_type]
#         if (pm_type == GON_PM_TYPE.CONSERVATIVE) or (pm_type == GON_PM_TYPE.BUY_AND_HOLD) or (pm_type == GON_PM_TYPE.LP_PROVIDER) :
#             avg_cash_out = None
#         else:
#             avg_cash_out = pd.DataFrame([{'investor_name': gon_investor.get_investor_name(),
#                                           'pct_yield_target': gon_investor._strat_buy_yield_pct} if gon_investor._pm_type == pm_type else {}
#                                      for gon_investor in gon_investors_retail_strategy]) \
#                             .dropna().set_index('investor_name').mean().values[0]
#         plot_pnl_score_card(strategy_pnl=strategy_pnl_on_pm, strategy_roi=strategy_roi.assign(avg_cash_out=avg_cash_out),
#                             gon_to_usdt_eod=gon_to_usdt_eod,
#                             title=f'Gon Participant Pnl Analytics Over time - {pm_type}')

plot_pnl_total_per_pm_group(strategy_pnl=strategy_pnl, gon_pm_types=GON_PM_TYPE.to_dict().keys())
sharpe_stats = compute_stats_per_pm_group(strategy_pnl=strategy_pnl, gon_pm_types=GON_PM_TYPE.to_dict().keys())

### get some gon dsitribution stats and plot on all actors:
plot_gonomics_distribution_plan(gonomics_engine=gonomics_engine, GON_total_supply=GON_total_supply, gon_to_usdt_eod=gon_to_usdt_eod,
                                community_map=pd.concat([pd.Series({i: gon_investors_pct_map[i]['pct']}) for i in gon_investors_pct_map.keys()]))
plot_gon_related_stats(gonomics_engine=gonomics_engine, GON_total_supply=GON_total_supply, gon_to_usdt_eod=gon_to_usdt_eod)