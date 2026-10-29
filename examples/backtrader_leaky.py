"""SMA trend filter in backtrader, with a classic look-ahead: the signal compares TOMORROW's close with the SMA.

self.data.close[1] is the NEXT bar. backtrader with preloaded data happily returns it, the backtest looks
great, and live trading cannot reproduce it.
"""
import backtrader as bt


class SmaTrend(bt.Strategy):
    params = (("period", 20),)

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None

    def next(self):
        if self.order:
            return
        price = self.data.close[1]          # look-ahead: next bar's close ([0] is today)
        if not self.position:
            if price > self.sma[0]:
                self.order = self.buy()
        elif price < self.sma[0]:
            self.order = self.close()


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaTrend)
    cerebro.run()
