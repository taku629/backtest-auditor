"""vectorbt moving-average crossover, as it often appears in a notebook. The entry condition also asks
for tomorrow's close to be higher: close.shift(-1) is the future."""
import vectorbt as vbt

close = vbt.YFData.download(["AAPL", "MSFT", "NVDA"], start="2018-01-01").get("Close")

fast = vbt.MA.run(close, 10)
slow = vbt.MA.run(close, 50)
up_tomorrow = close.shift(-1) > close                  # look-ahead
entries = fast.ma_crossed_above(slow) | (fast.ma_above(slow) & up_tomorrow)
exits = fast.ma_crossed_below(slow) | (close.shift(-1) < close * 0.98)

pf = vbt.Portfolio.from_signals(close, entries, exits, fees=0.001, freq="1D")
print(pf.total_return())
