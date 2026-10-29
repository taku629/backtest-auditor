"""SMA crossover in backtrader, point-in-time: only [0] and [-1] (today and yesterday) are read."""
import backtrader as bt


class SmaCross(bt.Strategy):
    params = (("fast", 20), ("slow", 100))

    def __init__(self):
        self.fast = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.slow = bt.indicators.SMA(self.data.close, period=self.p.slow)
        self.cross = bt.indicators.CrossOver(self.fast, self.slow)

    def next(self):
        if not self.position:
            if self.cross > 0 or (self.fast[0] > self.slow[0] and self.data.close[0] > self.data.close[-1]):
                self.buy()
        elif self.cross < 0:
            self.close()


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)
    cerebro.run()
