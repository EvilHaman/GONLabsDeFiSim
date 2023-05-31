import pandas as pd
from utilis.utils import setupLogger, compute_apy_stats

logger = setupLogger(__name__)
from utilis.constants import GON_PARTICIPANT_ACTION_TYPE

def run_simulation(rtn_daily, strat_contract, gonomics_engine, amm_pool, gon_investors_retail_strategy,
                   gon_lp_provider, gon_pre_seeders, gon_founders, sim_start_date='2019-12-31'):
    res = []
    gon_to_usdt_tracker = pd.Series({sim_start_date: amm_pool.reserve1 / amm_pool.reserve0})
    day_number = 0

    [gon_investor.on_start(strat_contract=strat_contract, amm_pool=amm_pool, datetime=sim_start_date) \
     for gon_investor in gon_investors_retail_strategy]

    apy_stats = compute_apy_stats(daily_rtn_series=rtn_daily.dropna())


    for date, daily_rtn in rtn_daily.dropna().iteritems():
        logger.debug(f'processing date: {date}...')

        ## compute daily return based on new TVL:
        new_tvl_usdt = strat_contract.get_strategy_tvl() * (1 + daily_rtn)
        pnl_usdt = new_tvl_usdt - strat_contract.get_strategy_tvl()

        ## the strategy contract is updated with the new pnl:
        performance_fee_usdt = strat_contract.update_tvl(pnl_usdt=pnl_usdt) if pnl_usdt != 0. else 0.

        if performance_fee_usdt > 0.:
            gon_burned = gonomics_engine.buy_and_burn(amount_usdt=performance_fee_usdt, amm_pool=amm_pool,
                                                      gas_fee_usdt=None, datetime=date)
            logger.debug(f'gon_burned: {gon_burned}...')

        ## distribute rewards to strategy participants and record them on their accounts:
        rewards = gonomics_engine.distribute_gon_rewards(strat_contract=strat_contract, amm_pool=amm_pool,
                                                         datetime=date)
        if len(rewards) > 0:
            strategy_gon_rewards, amm_rewards, gon_rewards_founders, gon_rewards_pre_seeders = rewards

            for strat_investor in gon_investors_retail_strategy + [gon_lp_provider]:
                if strategy_gon_rewards.get(strat_investor.get_investor_name(), 0.) > 0:
                    strat_investor.record_reward(amount=strategy_gon_rewards.get(strat_investor.get_investor_name()),
                                                 datetime=date, symbol='GON', symbol_pair='STRATEGY/GON',
                                                 action_type=GON_PARTICIPANT_ACTION_TYPE.RECORD_REWARDS_ON_STRATEGY)
                if amm_rewards['gon'].get(strat_investor.get_investor_name(), 0.) > 0:
                    ## NOTE that AMM/GON record is "just" record keeping and the amount is added in lp tokens and recorded as 'AMM/USDT'
                    strat_investor.record_reward(amount=amm_rewards['gon'].get(strat_investor.get_investor_name()),
                                                 datetime=date, symbol='GON', symbol_pair='AMM/GON',
                                                 action_type=GON_PARTICIPANT_ACTION_TYPE.RECORD_REWARDS_ON_LP)
                    strat_investor.record_reward(
                        amount=amm_rewards['liquidity_token'].get(strat_investor.get_investor_name()),
                        datetime=date, symbol='USDT', symbol_pair='AMM/USDT',
                        action_type=GON_PARTICIPANT_ACTION_TYPE.RECORD_REWARDS_ON_LP)

            ## for pre-seeders and founders, record rewards and add coins to their accounts:
            gon_pre_seeders._GON += gon_rewards_pre_seeders
            gon_founders._GON += gon_rewards_founders
            gon_pre_seeders.record_reward(amount=gon_rewards_pre_seeders,
                                          datetime=date, symbol='GON', symbol_pair='GONOMICS/GON',
                                          action_type=GON_PARTICIPANT_ACTION_TYPE.RECORD_REWARDS_ON_SEEDERS)
            gon_founders.record_reward(amount=gon_rewards_founders,
                                       datetime=date, symbol='GON', symbol_pair='GONOMICS/GON',
                                       action_type=GON_PARTICIPANT_ACTION_TYPE.RECORD_REWARDS_ON_FOUNDERS)

        [gon_investor.what_todo_today(strat_contract=strat_contract, amm_pool=amm_pool, datetime=date,
                                      start_apy_stats=apy_stats.loc[date],
                                      gon_to_usdt_tracker=gon_to_usdt_tracker) \
         for gon_investor in gon_investors_retail_strategy]

        ## TODO: NOTE: massive hack!! when GON\USDT trading on amm, I record the LP tokens and below add it to each investor trade log.
        ## the actual LP tokens are added to all liquidity providers on the spot.
        ## record LP creation based on any AMM activity:
        lp_tokens_to_record = amm_pool.get_liquidity_token_rewards_recorder()
        for strat_investor in gon_investors_retail_strategy + [gon_lp_provider]:
            number_of_lp_tokens = lp_tokens_to_record.get(strat_investor.get_investor_name(), 0.)
            if number_of_lp_tokens > 0:
                strat_investor.record_reward(amount=number_of_lp_tokens,
                                             datetime=date, symbol='USDT', symbol_pair='AMM/USDT',
                                             action_type=GON_PARTICIPANT_ACTION_TYPE.AMM_TRADE_FEES)
        amm_pool.reset_liquidity_token_rewards_recorder()

        ## mark to market the investors every 1 days:
        logger.debug(f'mark to market the investors...')
        for gon_investor in gon_investors_retail_strategy:
            gon_investor.mark_to_market(datetime=date, strat_contract=strat_contract, amm_pool=amm_pool)
            gon_investor.gather_analytics(datetime=date, strat_contract=strat_contract, amm_pool=amm_pool)
        gon_lp_provider.mark_to_market(datetime=date, strat_contract=strat_contract, amm_pool=amm_pool)

        ## gather some analytics:
        analytics = pd.Series({
            'daily_rtn': daily_rtn,
            'pnl_usdt': pnl_usdt,
            'tvl_usdt': strat_contract.get_strategy_tvl(),
            'gon_price': amm_pool.reserve1 / amm_pool.reserve0,
            'strat_index': strat_contract.get_index_price(),
            'today': pd.to_datetime(date),
        })
        res.append(analytics)
        gon_to_usdt_tracker.append(pd.Series({date: amm_pool.reserve1 / amm_pool.reserve0}))

    return res