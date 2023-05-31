# class SimpleAmm:
#     #simple auto market maker (amm) pool which works on k = x * y
#     def __init__(self, token_a_name, token_b_name, token_a_reserve, token_b_reserve):
#         self._token_a_name = token_a_name
#         self._token_b_name = token_b_name
#         self._token_a_reserve = token_a_reserve
#         self._token_b_reserve = token_b_reserve
#         self._liquidity_providers = {}
#
#     def add_liquidity(self, provider, token_a_amount, token_b_amount):
#         self._liquidity_providers[provider] = (token_a_amount, token_b_amount)
#         self._token_a_reserve += token_a_amount
#         self._token_b_reserve += token_b_amount
#
#
#     def _mint(self, to, value):
#         if self.liquidity_providers.get(to):
#             self.liquidity_providers[to] += value
#         else:
#             self.liquidity_providers[to] = value
#
#         self.total_supply += value
