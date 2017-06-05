# -*- coding: utf-8 -*-

from quantdigger import *
import timeit, math, copy
from MY_digger.riskctl import Riskctl

tt = {
    'max': [],
    'min': [],
    'N': []
    }

CTS = 'I.SHFE-1.Day'
RSK = 0.1


class DemoStrategy(Strategy):
    """ 策略 20 Days """


    def on_init(self, ctx):
        """初始化数据"""
        ctx.tt20 = Turtle(ctx.close, 20, 'tt20', ('r', 'g'))
        ctx.tt10 = Turtle(ctx.close, 10, 'tt10', ('y', 'b'))

        self.num_unit = 0
        self.max_unit = 4
        self.N = [0, ]
        self.N20 = 0.0
        risk = Riskctl()
        row = risk.csv_test(CTS.split('.')[0])
        assert len(row) == 1
        print row
        self._ratio = row[0][4]
        self._tick = row[0][6]
        self._vol = row[0][7]

        self._unit = 1.0
        self.breakpoint = []

    def on_bar(self, ctx):
        if ctx.curbar > 1:
            self.N.append(max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low)))

            #if ctx.curbar < 20:
                #print self.N[-1]

            if ctx.curbar == 20:
                for i in range(1, 20):
                    self.N20 += self.N[i]
                #print('N20 = %f, N = %f' % (self.N20, self.N20 / len(self.N)))
                self.N20 /= len(self.N)

        if ctx.curbar > 20:
            self.N20 = (self.N20 * 19 + self.N[-1]) / 20.0

            # 多头入市
            # 跳空高开
            if ctx.close[1] < ctx.tt20['max'][2] and ctx.open > ctx.tt20['max'][1] and self.num_unit == 0:
                _unit = self.unitctl(ctx)
                _open = copy.deepcopy(ctx.open)
                self.breakpoint.append([_open, _unit, 'buy'])
                ctx.buy(ctx.open, _unit)
                self.num_unit += 1
                self.display(ctx, 'buy')
            elif ctx.close[1] < ctx.tt20['max'][2] and ctx.close > ctx.tt20['max'][1] and self.num_unit == 0:
                _unit = self.unitctl(ctx)
                _close = copy.deepcopy(ctx.close)
                self.breakpoint.append([_close, _unit, 'buy'])
                ctx.buy(ctx.close, _unit)
                self.num_unit += 1
                self.display(ctx, 'buy')

            # 多头退市
            elif ctx.close[1] > ctx.tt10['min'][2] and ctx.close < ctx.tt10['min'][1] and \
                            self.num_unit > 0:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'buy'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units >= 0
                print ('sell_unit = %d' % all_units)
                ctx.sell(ctx.close, all_units)
                self.num_unit = 0
                self.display(ctx, 'sell')

            # 空头入市
            # 跳空低开
            if ctx.close[1] > ctx.tt20['min'][2] and ctx.open < ctx.tt20['min'][1] and \
                            self.num_unit == 0:
                _unit = self.unitctl(ctx)
                _open = copy.deepcopy(ctx.open)
                self.breakpoint.append([_open, _unit, 'short'])
                ctx.short(ctx.open, _unit)
                self.num_unit -= 1
                self.display(ctx, 'short')
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.close < ctx.tt20['min'][1] and \
                            self.num_unit == 0:
                _unit = self.unitctl(ctx)
                _close = copy.deepcopy(ctx.close)
                self.breakpoint.append([_close, _unit, 'short'])
                ctx.short(ctx.close, _unit)
                self.num_unit -= 1
                self.display(ctx, 'short')

            # 空头退市
            elif ctx.close[1] < ctx.tt10['max'][2] and ctx.close > ctx.tt10['max'][1] and \
                            self.num_unit < 0:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'short'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units >= 0
                print ('cover_unit = %d' % all_units)
                ctx.cover(ctx.close, all_units)
                self.num_unit = 0
                self.display(ctx, 'cover')

            # 多头加仓
            if 0 < self.num_unit < 4:
                # 跳空高开
                if (self.breakpoint[-1][0] + self.N20 * 0.5) < ctx.open:
                    _unit = self.unitctl(ctx)
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'buy'])
                    ctx.buy(ctx.open, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')
                elif (self.breakpoint[-1][0] + self.N20 * 0.5) < ctx.close:
                    _unit = self.unitctl(ctx)
                    _close = copy.deepcopy(ctx.close)
                    self.breakpoint.append([_close, _unit, 'buy'])
                    ctx.buy(ctx.close, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')
                # 多头止损
                elif (self.breakpoint[-1][0] - self.N20 * 2) > ctx.close:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)
                    ctx.sell(ctx.close, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'sell')

            # 空头加仓
            if -4 < self.num_unit < 0:
                # 跳空低开
                if (self.breakpoint[-1][0] - self.N20 * 0.5) > ctx.open:
                    _unit = self.unitctl(ctx)
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'short'])
                    ctx.short(ctx.open, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')
                elif (self.breakpoint[-1][0] - self.N20 * 0.5) > ctx.close:
                    _unit = self.unitctl(ctx)
                    _close = copy.deepcopy(ctx.close)
                    self.breakpoint.append([_close, _unit, 'short'])
                    ctx.short(ctx.close, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')
                # 空头止损
                elif (self.breakpoint[-1][0] + self.N20 * 2) < ctx.close:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units >= 0
                    print ('cover_unit = %d' % all_units)
                    ctx.cover(ctx.close, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'cover')

    def on_symbol(self, ctx):
        return

    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  20 Days **************')
        print ('N20 = %f, N = %f' % (self.N20, self.N[-1]))

        print('%s %s; Number of Units = %d' % (name, ctx.symbol, self.num_unit))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell':
            tag = 'long'
        elif name == 'short' or name == 'cover':
            tag = 'short'
        else:
            tag = 'unkonwn'
        print (ctx.position(tag, 'BB.SHFE'))

    def unitctl(self, ctx, name='nameless'):
        cash = ctx.cash()
        _ratio = float(self._ratio)
        _vol = int(self._vol)
        _unit = math.floor((cash * RSK * _ratio) / (self.N20 * _vol))
        print ('_unit = %d, real_unit = %f' % (_unit, (cash * RSK * _ratio) / (self.N20 * _vol)))
        return int(_unit)


class DemoStrategy2(Strategy):
    """ 策略 55 Days """

    def on_init(self, ctx):
        """初始化数据"""
        ctx.tt55 = Turtle(ctx.close, 55, 'tt20', ('y', 'b'))
        ctx.tt20 = Turtle(ctx.close, 20, 'tt20', ('r', 'g'))
        #ctx.tt10 = Turtle(ctx.close, 10, 'tt10', ('y', 'b'))

        self.num_unit = 0
        self.max_unit = 4
        self.N = [0, ]
        self.N20 = 0.0
        risk = Riskctl()
        row = risk.csv_test(CTS.split('.')[0])
        assert len(row) == 1
        print row
        self._ratio = row[0][4]
        self._tick = row[0][6]
        self._vol = row[0][7]

        self._unit = 1.0
        self.breakpoint = []

    def on_bar(self, ctx):
        if ctx.curbar > 1:
            self.N.append(max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low)))

            #if ctx.curbar < 20:
                #print self.N[-1]

            if ctx.curbar == 20:
                for i in range(1, 20):
                    self.N20 += self.N[i]
                self.N20 /= len(self.N)

        if ctx.curbar > 55:
            self.N20 = (self.N20 * 19 + self.N[-1]) / 20.0

            # 多头入市
            # 跳空高开
            if ctx.close[1] < ctx.tt55['max'][2] and ctx.open > ctx.tt55['max'][1] and self.num_unit == 0:
                _unit = self.unitctl(ctx)
                _open = copy.deepcopy(ctx.open)
                self.breakpoint.append([_open, _unit, 'buy'])
                ctx.buy(ctx.open, _unit)
                self.num_unit += 1
                self.display(ctx, 'buy')
            elif ctx.close[1] < ctx.tt55['max'][2] and ctx.close > ctx.tt55['max'][1] and self.num_unit == 0:
                _unit = self.unitctl(ctx)
                _close = copy.deepcopy(ctx.close)
                self.breakpoint.append([_close, _unit, 'buy'])
                ctx.buy(ctx.close, _unit)
                self.num_unit += 1
                self.display(ctx, 'buy')

            # 多头退市
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.close < ctx.tt20['min'][1] and \
                            self.num_unit > 0:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'buy'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units >= 0
                print ('sell_unit = %d' % all_units)
                ctx.sell(ctx.close, all_units)
                self.num_unit = 0
                self.display(ctx, 'sell')

            # 空头入市
            # 跳空低开
            if ctx.close[1] > ctx.tt55['min'][2] and ctx.open < ctx.tt55['min'][1] and \
                            self.num_unit == 0:
                _unit = self.unitctl(ctx)
                _open = copy.deepcopy(ctx.open)
                self.breakpoint.append([_open, _unit, 'short'])
                ctx.short(ctx.open, _unit)
                self.num_unit -= 1
                self.display(ctx, 'short')
            elif ctx.close[1] > ctx.tt55['min'][2] and ctx.close < ctx.tt55['min'][1] and \
                            self.num_unit == 0:
                _unit = self.unitctl(ctx)
                _close = copy.deepcopy(ctx.close)
                self.breakpoint.append([_close, _unit, 'short'])
                ctx.short(ctx.close, _unit)
                self.num_unit -= 1
                self.display(ctx, 'short')

            # 空头退市
            elif ctx.close[1] < ctx.tt20['max'][2] and ctx.close > ctx.tt20['max'][1] and \
                            self.num_unit < 0:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'short'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units >= 0
                print ('cover_unit = %d' % all_units)
                ctx.cover(ctx.close, all_units)
                self.num_unit = 0
                self.display(ctx, 'cover')

            # 多头加仓
            if 0 < self.num_unit < 4:
                # 跳空高开
                if (self.breakpoint[-1][0] + self.N20 * 0.5) < ctx.open:
                    _unit = self.unitctl(ctx)
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'buy'])
                    ctx.buy(ctx.open, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')
                elif (self.breakpoint[-1][0] + self.N20 * 0.5) < ctx.close:
                    _unit = self.unitctl(ctx)
                    _close = copy.deepcopy(ctx.close)
                    self.breakpoint.append([_close, _unit, 'buy'])
                    ctx.buy(ctx.close, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')
                # 多头止损
                elif (self.breakpoint[-1][0] - self.N20 * 2) > ctx.close:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)
                    ctx.sell(ctx.close, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'sell')

            # 空头加仓
            if -4 < self.num_unit < 0:
                # 跳空低开
                if (self.breakpoint[-1][0] - self.N20 * 0.5) > ctx.open:
                    _unit = self.unitctl(ctx)
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'short'])
                    ctx.short(ctx.open, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')
                elif (self.breakpoint[-1][0] - self.N20 * 0.5) > ctx.close:
                    _unit = self.unitctl(ctx)
                    _close = copy.deepcopy(ctx.close)
                    self.breakpoint.append([_close, _unit, 'short'])
                    ctx.short(ctx.close, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')
                # 空头止损
                elif (self.breakpoint[-1][0] + self.N20 * 2) < ctx.close:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units >= 0
                    print ('cover_unit = %d' % all_units)
                    ctx.cover(ctx.close, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'cover')

    def on_symbol(self, ctx):
        return

    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  55 Days **************')
        print ('N20 = %f, N = %f' % (self.N20, self.N[-1]))

        print('%s %s; Number of Units = %d' % (name, ctx.symbol, self.num_unit))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell':
            tag = 'long'
        elif name == 'short' or name == 'cover':
            tag = 'short'
        else:
            tag = 'unkonwn'
        print (ctx.position(tag, 'BB.SHFE'))

    def unitctl(self, ctx, name='nameless'):
        cash = ctx.cash()
        _ratio = float(self._ratio)
        _vol = int(self._vol)
        _unit = math.floor((cash * RSK * _ratio) / (self.N20 * _vol))
        print ('_unit = %d, real_unit = %f' % (_unit, (cash * RSK * _ratio) / (self.N20 * _vol)))
        return int(_unit)


if __name__ == '__main__':
    from quantdigger.digger import finance, plotting
    import MY_digger.myplot

    set_symbols([CTS])
    profile = add_strategy([DemoStrategy('20 Days')], {'capital': 100000})
    profile2 = add_strategy([DemoStrategy2('55 Days')], {'capital': 100000})

    start = timeit.default_timer()
    run()
    stop = timeit.default_timer()
    print "运行耗时: %d秒" % (stop - start)

    # 打印组合1的统计信息
    curve = finance.create_equity_curve(profile.all_holdings())
    print '20 Days', finance.summary_stats(curve, 252 * 4 * 60)
    curve2 = finance.create_equity_curve(profile2.all_holdings())
    print '55 Days', finance.summary_stats(curve2, 252 * 4 * 60)

    # K-line
    MY_digger.myplot.plot_strategy(profile.data(0), profile.technicals(0), profile.deals(0), curve.equity)
    MY_digger.myplot.plot_strategy(profile2.data(0), profile2.technicals(0), profile2.deals(0), curve2.equity)

    # Equity-curve
    #myplot.plot_curves([curve.equity], colors=['r'], names=[profile.name(0)])
    MY_digger.myplot.plot_curves([curve.equity, curve2.equity], colors=['r', 'b'], names=[profile.name(0), profile2.name(0)])

    MY_digger.myplot.show()

    r = Riskctl()
    m = r.csv_test(CTS.split('.')[0])
    assert (len(m) == 1)
