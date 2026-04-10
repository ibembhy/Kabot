import sys
sys.path.insert(0, 'src')

# From the 30-day backtest:
# Total PnL = $209.04 over 31 trading days with 1 contract per trade
DAILY_PNL_PER_CONTRACT = 209.04 / 31.0   # ~$6.74/day per contract

AVG_PRICE  = 0.50   # avg entry price per contract
FRACTION   = 0.08   # 8% of bankroll targeted per trade
MAX_CONS   = 2      # conservative: max_contracts_per_order (current live setting)
MAX_OPT    = 4      # optimistic: max_contracts_per_market


def contracts_at(bankroll: float, cap: int) -> int:
    c = int((bankroll * FRACTION) // AVG_PRICE)
    return max(1, min(c, cap))


def grow_one_month(bankroll: float, cap: int, days: int = 30) -> float:
    for _ in range(days):
        bankroll += DAILY_PNL_PER_CONTRACT * contracts_at(bankroll, cap)
    return bankroll


print("=" * 60)
print(f"  Starting bankroll: $45.00")
print(f"  Win rate: 62.2%  |  30-day historical data")
print(f"  Conservative = 2 contracts/order (current live setting)")
print(f"  Optimistic   = 4 contracts/market (if all fills land)")
print("=" * 60)
print(f"{'Month':<8} {'Conservative':>15} {'Optimistic':>14}")
print("-" * 40)

bk_cons = 45.0
bk_opt  = 45.0

for month in range(1, 7):
    bk_cons = grow_one_month(bk_cons, MAX_CONS)
    bk_opt  = grow_one_month(bk_opt,  MAX_OPT)
    c_cons  = contracts_at(bk_cons, MAX_CONS)
    c_opt   = contracts_at(bk_opt,  MAX_OPT)
    print(
        f"  {month:<6}"
        f"  ${bk_cons:>9.2f} ({c_cons}c)"
        f"   ${bk_opt:>9.2f} ({c_opt}c)"
    )

print("-" * 40)
print(f"\n  (c) = contracts per trade at end of that month")
print(f"  Assumes same win rate and trade frequency each month.")
print(f"  Real results will vary — treat as directional, not a guarantee.")
