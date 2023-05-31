#
# class Exchange:
#
#     def __init__(self, token0_name: str, token1_name: str, name: str, symbol: str) -> None:
#         self.token0 = token0_name
#         self.token1 = token1_name
#         self.reserve0 = 0
#         self.reserve1 = 0
#         self.fee = 0
#
#         self.name = name
#         self.symbol = symbol
#         self.liquidity_providers = {}
#         self.total_supply = 0
#
#     def quote(self, amount0, reserve0=None, reserve1=None):
#         """
#         Given some amount of an asset and pair reserves, returns an equivalent amount of
#         the other asset
#         """
#         assert amount0 > 0, 'UniswapV2Library: INSUFFICIENT_AMOUNT'
#         reserve0_to_use = reserve0 if reserve0 is not None else self.reserve0
#         reserve1_to_use = reserve1 if reserve1 is not None else self.reserve1
#         assert reserve0_to_use > 0 and reserve1_to_use > 0, 'UniswapV2Library: INSUFFICIENT_LIQUIDITY'
#         return (amount0 * reserve1_to_use) / reserve0_to_use;
#
#     def _add_liquidity(self, balance0, balance1):
#         """
#         You always need to add liquidity to both types of coins
#         """
#         if self.reserve0 == 0 and self.reserve1 == 0:
#             # initializing pool
#             amount0 = balance0
#             amount1 = balance1
#         else:
#             balance1Optimal = self.quote(balance0, self.reserve0, self.reserve1)
#             if balance1Optimal <= balance1:
#                 amount0 = balance0
#                 amount1 = balance1Optimal
#             else:
#                 balance0Optimal = self.quote(balance1, self.reserve1, self.reserve0)
#                 assert balance0Optimal <= balance0
#                 amount0 = balance0Optimal
#                 amount1 = balance1
#
#         self.reserve0 += amount0
#         self.reserve1 += amount1
#
#     def get_amount_out(self, amount_in):
#         """
#         Given an input amount of an asset and pair reserves, returns the maximum output amount of the
#         other asset
#
#         (reserve0 + amount_in_with_fee) * (reserve1 - amount_out) = reserve1 * reserve0
#         """
#         assert amount_in > 0, 'UniswapV2Library: INSUFFICIENT_INPUT_AMOUNT'
#         assert self.reserve0 > 0 and self.reserve1 > 0, 'UniswapV2Library: INSUFFICIENT_LIQUIDITY'
#
#         amount_in_with_fee = amount_in * 1000  # disconsidering the fee here: amount_in * 997
#         numerator = amount_in_with_fee * self.reserve1
#         denominator = self.reserve0 * 1000 + amount_in_with_fee
#         amount_out = numerator / denominator
#
#         return amount_out
#
#     def swapExactTokensForTokens(self, amount0_in, amount1_out_min):
#         amount0_out = 0
#         amount1_out = self.get_amount_out(amount0_in)
#         assert amount1_out >= amount1_out_min, 'UniswapV2Router: INSUFFICIENT_OUTPUT_AMOUNT'
#
#         assert amount1_out > 0, 'UniswapV2: INSUFFICIENT_OUTPUT_AMOUNT'
#         assert amount0_out < self.reserve0 and amount1_out < self.reserve1, 'UniswapV2: INSUFFICIENT_LIQUIDITY'
#
#         balance0 = self.reserve0 + amount0_in - amount0_out
#         balance1 = self.reverve1 - amount1_out
#         balance0_adjusted = balance0 * 1000
#         balance1_adjusted = balance1 * 1000
#         assert balance0_adjusted * balance1_adjusted == self.reserve0 * self.reserve1 * 1000 ** 2, 'UniswapV2: K'
#
#         self.reserve0 = balance0
#         self.reserve1 = balance1
#
#         return amount1_out
#
# gon_amm_v2 = Exchange(token0_name='GON', token1_name= 'USDT', name= 'AMMV2', symbol='GON/USDT' )
# gon_amm_v2._add_liquidity(balance0=10, balance1=1000)
# gon_amm_v2.get_amount_out(10000)
# gon_amm_v2.quote(amount0=2, reserve0=None, reserve1=None)
# gon_amm_v2.swapExactTokensForTokens(10, 0.1)
