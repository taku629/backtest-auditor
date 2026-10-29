"""vectorbt RSI mean reversion, point-in-time: signals use only data up to each bar."""
import vectorbt as vbt


def get_signals(close, window=14, lower=30, upper=55):
    rsi = vbt.RSI.run(close, window=window)
    entries = rsi.rsi_crossed_below(lower)
    exits = rsi.rsi_crossed_above(upper)
    return entries, exits


if __name__ == "__main__":
    close = vbt.YFData.download(["AAPL", "MSFT"]).get("Close")
    entries, exits = get_signals(close)
    pf = vbt.Portfolio.from_signals(close, entries, exits)
    print(pf.stats())
