# -*- coding: utf-8 -*-

from base import *
from quantdigger import *
import timeit, math, copy
from config import Default


CAPITAL = Default.get_capital()


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
        self.n20 = 0.0
        # risk = Riskctl()
        # row = risk.csv_test(CTS.split('.')[0])
        # assert len(row) == 1
        # print row
        # self._ratio = row[0][4]
        # self._tick = row[0][6]
        # self._vol = row[0][7]

        self._unit = 1.0
        self.breakpoint = []

        self.risk_ctl = Risk_ctl()

        self.candicates = []

    def on_symbol(self, ctx):
        self.candicates = 'I'

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
            cash = ctx.cash()

            # force cover
            if cash < 0:
                print ('### Current print = %f' % cash)
                if self.num_unit > 0:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.low, ctx.ttout['min'][1])
                    ctx.sell(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'sell-force')
                elif self.num_unit < 0:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.low, ctx.ttout['max'][1])
                    ctx.cover(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'cover-force')

            # 多头入市
            # 跳空高开
            if ctx.close[1] < ctx.tt20['max'][2] and ctx.open > ctx.tt20['max'][1] and self.num_unit == 0:
                self.n20 = self.N20

                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                if _unit > 0:
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'buy'])
                    ctx.buy(ctx.open, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')
            elif ctx.close[1] < ctx.tt20['max'][2] and ctx.high > ctx.tt20['max'][1] and self.num_unit == 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                if _unit > 0:
                    _price = copy.deepcopy(ctx.tt20['max'][1])
                    self.breakpoint.append([_price, _unit, 'buy'])
                    ctx.buy(_price, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')

            # 多头退市
            elif ctx.close[1] > ctx.tt10['min'][2] and ctx.low < ctx.tt10['min'][1] and \
                            self.num_unit > 0:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'buy'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units > 0
                print ('sell_unit = %d' % all_units)
                _price = self._cmp(ctx.high, ctx.low, ctx.tt10['min'][1])
                ctx.sell(_price, all_units)
                self.num_unit = 0
                self.display(ctx, 'sell')

            # 空头入市
            # 跳空低开
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.open < ctx.tt20['min'][1] and \
                    self.num_unit == 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                if _unit > 0:
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'short'])
                    ctx.short(ctx.open, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.low < ctx.tt20['min'][1] and \
                    self.num_unit == 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                if _unit > 0:
                    _price = copy.deepcopy(ctx.tt20['min'][1])
                    self.breakpoint.append([_price, _unit, 'short'])
                    ctx.short(_price, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')

            # 空头退市
            elif ctx.close[1] < ctx.tt10['max'][2] and ctx.high > ctx.tt10['max'][1] and \
                    self.num_unit < 0:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'short'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units > 0
                print ('cover_unit = %d' % all_units)
                _price = self._cmp(ctx.high, ctx.low, ctx.tt10['max'][1])
                ctx.cover(_price, all_units)
                self.num_unit = 0
                self.display(ctx, 'cover')

            # 多头加仓
            elif 0 < self.num_unit < 4:
                # 跳空高开
                if (self.breakpoint[-1][0] + self.n20 * 0.5) < ctx.open:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint.append([_open, _unit, 'buy'])
                        ctx.buy(ctx.open, _unit)
                        self.num_unit += 1
                        self.display(ctx, 'buy')
                elif (self.breakpoint[-1][0] + self.n20 * 0.5) < ctx.high:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    if _unit > 0:
                        _price = round(self.breakpoint[-1][0] + self.n20 * 0.5, 2)
                        self.breakpoint.append([_price, _unit, 'buy'])
                        ctx.buy(_price, _unit)
                        self.num_unit += 1
                        self.display(ctx, 'buy')
                # 多头止损
                elif (self.breakpoint[-1][0] - self.n20 * 2) > ctx.low:
                    all_units = 0
                    p = round(self.breakpoint[-1][0] - self.n20 * 2, 2)
                    _price = self._cmp(ctx.high, ctx.low, p)
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)

                    ctx.sell(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'sell')

            # 空头加仓
            elif -4 < self.num_unit < 0:
                # 跳空低开
                if (self.breakpoint[-1][0] - self.n20 * 0.5) > ctx.open:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint.append([_open, _unit, 'short'])
                        ctx.short(ctx.open, _unit)
                        self.num_unit -= 1
                        self.display(ctx, 'short')
                elif (self.breakpoint[-1][0] - self.n20 * 0.5) > ctx.low:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    if _unit > 0:
                        _price = round(self.breakpoint[-1][0] - self.n20 * 0.5, 2)
                        self.breakpoint.append([_price, _unit, 'short'])
                        ctx.short(_price, _unit)
                        self.num_unit -= 1
                        self.display(ctx, 'short')
                # 空头止损
                elif (self.breakpoint[-1][0] + self.n20 * 2) < ctx.high:
                    all_units = 0
                    p = round(self.breakpoint[-1][0] + self.n20 * 2, 2)
                    _price = self._cmp(ctx.high, ctx.low, p)
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    ctx.cover(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'cover')

    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  20 Days  **************')
        print ('N20 = %f, N = %f' % (self.n20, self.N[-1]))

        print('%s %s; Number of Units = %d' % (name, ctx.symbol, self.num_unit))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        print (ctx.position(tag, 'BB.SHFE'))

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]


class DemoStrategy2(Strategy):
    """ 策略 55 Days """

    def on_init(self, ctx):
        """初始化数据"""
        ctx.tt55 = Turtle(ctx.close, 55, 'tt55', ('y', 'b'))
        ctx.tt20 = Turtle(ctx.close, 20, 'tt20', ('r', 'g'))
        #ctx.tt10 = Turtle(ctx.close, 10, 'tt10', ('y', 'b'))

        self.num_unit = 0
        self.max_unit = 4
        self.N = [0, ]
        self.N20 = 0.0
        self.n20 = 0.0

        self._unit = 1.0
        self.breakpoint = []
        self.risk_ctl = Risk_ctl()

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
            cash = ctx.cash()

            # force cover
            if cash < 0:
                print ('### Current print = %f' % cash)
                if self.num_unit > 0:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.low, ctx.ttout['min'][1])
                    ctx.sell(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'sell-force')
                elif self.num_unit < 0:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.low, ctx.ttout['max'][1])
                    ctx.cover(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'cover-force')

            # 多头入市
            # 跳空高开
            if ctx.close[1] < ctx.tt55['max'][2] and ctx.open > ctx.tt55['max'][1] and self.num_unit == 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                _open = copy.deepcopy(ctx.open)
                self.breakpoint.append([_open, _unit, 'buy'])
                ctx.buy(ctx.open, _unit)
                self.num_unit += 1
                self.display(ctx, 'buy')
            elif ctx.close[1] < ctx.tt55['max'][2] and ctx.high > ctx.tt55['max'][1] and self.num_unit == 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                _price = copy.deepcopy(ctx.tt55['max'][1])
                self.breakpoint.append([_price, _unit, 'buy'])
                ctx.buy(_price, _unit)
                self.num_unit += 1
                self.display(ctx, 'buy')

            # 多头退市
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.low < ctx.tt20['min'][1] and \
                            self.num_unit > 0:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'buy'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units > 0
                print ('sell_unit = %d' % all_units)
                _price = self._cmp(ctx.high, ctx.low, ctx.tt20['min'][1])
                ctx.sell(_price, all_units)
                self.num_unit = 0
                self.display(ctx, 'sell')

            # 空头入市
            # 跳空低开
            elif ctx.close[1] > ctx.tt55['min'][2] and ctx.open < ctx.tt55['min'][1] and \
                    self.num_unit == 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                _open = copy.deepcopy(ctx.open)
                self.breakpoint.append([_open, _unit, 'short'])
                ctx.short(ctx.open, _unit)
                self.num_unit -= 1
                self.display(ctx, 'short')
            elif ctx.close[1] > ctx.tt55['min'][2] and ctx.low < ctx.tt55['min'][1] and \
                    self.num_unit == 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                _price = copy.deepcopy(ctx.tt55['min'][1])
                self.breakpoint.append([_price, _unit, 'short'])
                ctx.short(_price, _unit)
                self.num_unit -= 1
                self.display(ctx, 'short')

            # 空头退市
            elif ctx.close[1] < ctx.tt20['max'][2] and ctx.high > ctx.tt20['max'][1] and \
                            self.num_unit < 0:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'short'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units > 0
                print ('cover_unit = %d' % all_units)
                _price = self._cmp(ctx.high, ctx.low, ctx.tt20['max'][1])
                ctx.cover(_price, all_units)
                self.num_unit = 0
                self.display(ctx, 'cover')

            # 多头加仓
            elif 0 < self.num_unit < 4:
                # 跳空高开
                if (self.breakpoint[-1][0] + self.n20 * 0.5) < ctx.open:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'buy'])
                    ctx.buy(ctx.open, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')
                elif (self.breakpoint[-1][0] + self.n20 * 0.5) < ctx.high:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    _price = round(self.breakpoint[-1][0] + self.n20 * 0.5, 2)
                    self.breakpoint.append([_price, _unit, 'buy'])
                    ctx.buy(_price, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')
                # 多头止损
                elif (self.breakpoint[-1][0] - self.n20 * 2) > ctx.low:
                    all_units = 0
                    p = round(self.breakpoint[-1][0] - self.n20 * 2, 2)
                    _price = self._cmp(ctx.high, ctx.low, p)
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)

                    ctx.sell(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'sell')

            # 空头加仓
            elif -4 < self.num_unit < 0:
                # 跳空低开
                if (self.breakpoint[-1][0] - self.n20 * 0.5) > ctx.open:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'short'])
                    ctx.short(ctx.open, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')
                elif (self.breakpoint[-1][0] - self.n20 * 0.5) > ctx.low:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    _price = round(self.breakpoint[-1][0] - self.n20 * 0.5, 2)
                    self.breakpoint.append([_price, _unit, 'short'])
                    ctx.short(_price, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')
                # 空头止损
                elif (self.breakpoint[-1][0] + self.n20 * 2) < ctx.high:
                    all_units = 0
                    p = round(self.breakpoint[-1][0] + self.n20 * 2, 2)
                    _price = self._cmp(ctx.high, ctx.low, p)
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    ctx.cover(_price, all_units)
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
        print ('N20 = %f, N = %f' % (self.n20, self.N[-1]))

        print('%s %s; Number of Units = %d' % (name, ctx.symbol, self.num_unit))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        print (ctx.position(tag, 'BB.SHFE'))

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]


class DemoStrategy3(Strategy):
    """ 策略 20 Days + Predict"""


    def on_init(self, ctx):
        """初始化数据"""
        ctx.tt20 = Turtle(ctx.close, 20, 'tt20', ('r', 'g'))
        ctx.tt10 = Turtle(ctx.close, 10, 'tt10', ('y', 'b'))

        self.num_unit = 0
        self.max_unit = 4
        self.N = [0, ]
        self.N20 = 0.0
        self.n20 = 0.0

        self._unit = 1.0
        self.breakpoint = []
        self.predict = Predict_module()
        self.risk_ctl = Risk_ctl()

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
            cash = ctx.cash()

            # force cover
            if cash < 0:
                print ('### Current print = %f' % cash)
                if self.num_unit > 0:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.low, ctx.ttout['min'][1])
                    ctx.sell(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'sell-force')
                elif self.num_unit < 0:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.low, ctx.ttout['max'][1])
                    ctx.cover(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'cover-force')

            # 多头入市
            # 跳空高开
            if ctx.close[1] < ctx.tt20['max'][2] and ctx.open > ctx.tt20['max'][1] and self.num_unit == 0\
                    and self.predict.signal(ctx.curbar) >= 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                _open = copy.deepcopy(ctx.open)
                self.breakpoint.append([_open, _unit, 'buy'])
                ctx.buy(ctx.open, _unit)
                self.num_unit += 1
                self.display(ctx, 'buy')
            elif ctx.close[1] < ctx.tt20['max'][2] and ctx.high > ctx.tt20['max'][1] and self.num_unit == 0 \
                    and self.predict.signal(ctx.curbar) >= 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                _price = copy.deepcopy(ctx.tt20['max'][1])
                self.breakpoint.append([_price, _unit, 'buy'])
                ctx.buy(_price, _unit)
                self.num_unit += 1
                self.display(ctx, 'buy')

            # 多头退市
            elif ctx.close[1] > ctx.tt10['min'][2] and ctx.low < ctx.tt10['min'][1] and \
                    self.num_unit > 0:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'buy'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units > 0
                print ('sell_unit = %d' % all_units)
                _price = self._cmp(ctx.high, ctx.low, ctx.tt10['min'][1])
                ctx.sell(_price, all_units)
                self.num_unit = 0
                self.display(ctx, 'sell')

            # 空头入市
            # 跳空低开
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.open < ctx.tt20['min'][1] and self.num_unit == 0 and \
                    self.predict.signal(ctx.curbar) <= 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                _open = copy.deepcopy(ctx.open)
                self.breakpoint.append([_open, _unit, 'short'])
                ctx.short(ctx.open, _unit)
                self.num_unit -= 1
                self.display(ctx, 'short')
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.low < ctx.tt20['min'][1] and self.num_unit == 0 and \
                    self.predict.signal(ctx.curbar) <= 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                _price = copy.deepcopy(ctx.tt20['min'][1])
                self.breakpoint.append([_price, _unit, 'short'])
                ctx.short(_price, _unit)
                self.num_unit -= 1
                self.display(ctx, 'short')

            # 空头退市
            elif ctx.close[1] < ctx.tt10['max'][2] and ctx.high > ctx.tt10['max'][1] and \
                            self.num_unit < 0:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'short'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units > 0
                print ('cover_unit = %d' % all_units)
                _price = self._cmp(ctx.high, ctx.low, ctx.tt10['max'][1])
                ctx.cover(_price, all_units)
                self.num_unit = 0
                self.display(ctx, 'cover')

            # 多头加仓
            elif 0 < self.num_unit < 4:
                # 跳空高开
                if (self.breakpoint[-1][0] + self.n20 * 0.5) < ctx.open \
                        and self.predict.signal(ctx.curbar) >= 0:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'buy'])
                    ctx.buy(ctx.open, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')
                elif (self.breakpoint[-1][0] + self.n20 * 0.5) < ctx.high \
                        and self.predict.signal(ctx.curbar) >= 0:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    _price = round(self.breakpoint[-1][0] + self.n20 * 0.5, 2)
                    self.breakpoint.append([_price, _unit, 'buy'])
                    ctx.buy(_price, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')
                # 多头止损
                elif (self.breakpoint[-1][0] - self.n20 * 2) > ctx.low:
                    all_units = 0
                    p = round(self.breakpoint[-1][0] - self.n20 * 2, 2)
                    _price = self._cmp(ctx.high, ctx.low, p)
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)

                    ctx.sell(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'sell')

            # 空头加仓
            elif -4 < self.num_unit < 0:
                # 跳空低开
                if (self.breakpoint[-1][0] - self.n20 * 0.5) > ctx.open \
                        and self.predict.signal(ctx.curbar) <= 0:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'short'])
                    ctx.short(ctx.open, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')
                elif (self.breakpoint[-1][0] - self.n20 * 0.5) > ctx.low \
                        and self.predict.signal(ctx.curbar) <= 0:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    _price = round(self.breakpoint[-1][0] - self.n20 * 0.5, 2)
                    self.breakpoint.append([_price, _unit, 'short'])
                    ctx.short(_price, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')
                # 空头止损
                elif (self.breakpoint[-1][0] + self.n20 * 2) < ctx.high:
                    all_units = 0
                    p = round(self.breakpoint[-1][0] + self.n20 * 2, 2)
                    _price = self._cmp(ctx.high, ctx.low, p)
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    ctx.cover(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'cover')

    def on_symbol(self, ctx):
        return

    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('*********  20 Days + Predict  **********')
        print ('N20 = %f, N = %f' % (self.n20, self.N[-1]))

        print('%s %s; Number of Units = %d' % (name, ctx.symbol, self.num_unit))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        print (ctx.position(tag, 'BB.SHFE'))

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]


class MainStrategy(Strategy):
    """ 策略 用户自定义倍率 """


    def on_init(self, ctx):
        """初始化数据"""

        self.tt = Default.get_tt()

        ctx.ttin = Turtle(ctx.close, self.tt['IN'], 'ttin', ('r', 'g'))
        ctx.ttout = Turtle(ctx.close, self.tt['OUT'], 'ttout', ('y', 'b'))

        self.num_unit = 0
        self.max_unit = Default.get_maxunit()
        self.N = [0, ]
        self.N20 = 0.0
        self.n20 = 0.0

        self.oc = Operation_cycle()
        self._unit = 1.0
        self.breakpoint = []

        self.predict = Predict_module()

        self.risk_ctl = Risk_ctl()

    def on_bar(self, ctx):
        _maxunit = self.max_unit * -1
        maxunit = self.max_unit
        if ctx.curbar > 1:
            self.N.append(max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low)))

            #if ctx.curbar < 20:
                #print self.N[-1]

            if ctx.curbar == self.tt['N']:
                for i in range(1, self.tt['N']):
                    self.N20 += self.N[i]
                #print('N20 = %f, N = %f' % (self.N20, self.N20 / len(self.N)))
                self.N20 /= len(self.N)

        if ctx.curbar > self.tt['N']:
            self.N20 = (self.N20 * (self.tt['N'] - 1) + self.N[-1]) / self.tt['N'] * 1.0
            op_cycle = self.oc.counter()
            cash = ctx.cash()

            # force cover
            if cash < 0:
                print ('### Current print = %f' % cash)
                if self.num_unit > 0:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.low, ctx.ttout['min'][1])
                    ctx.sell(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'sell-force')
                elif self.num_unit < 0:
                    all_units = 0
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.low, ctx.ttout['max'][1])
                    ctx.cover(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'cover-force')

            # 多头入市
            # 跳空高开
            if ctx.close[1] < ctx.ttin['max'][2] and ctx.open > ctx.ttin['max'][1] and self.num_unit == 0 and\
                    op_cycle\
                    and self.predict.signal(ctx.curbar) >= 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                if _unit > 0:
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'buy'])
                    ctx.buy(ctx.open, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')
            elif ctx.close[1] < ctx.ttin['max'][2] and ctx.high > ctx.ttin['max'][1] and self.num_unit == 0 and\
                    op_cycle\
                    and self.predict.signal(ctx.curbar) >= 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                if _unit > 0:
                    _price = copy.deepcopy(ctx.ttin['max'][1])
                    self.breakpoint.append([_price, _unit, 'buy'])
                    ctx.buy(_price, _unit)
                    self.num_unit += 1
                    self.display(ctx, 'buy')

            # 多头退市
            elif ctx.close[1] > ctx.ttout['min'][2] and ctx.low < ctx.ttout['min'][1] and \
                    self.num_unit > 0 and\
                    op_cycle:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'buy'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units > 0
                print ('sell_unit = %d' % all_units)
                _price = self._cmp(ctx.high, ctx.low, ctx.ttout['min'][1])
                ctx.sell(_price, all_units)
                self.num_unit = 0
                self.display(ctx, 'sell')

            # 空头入市
            # 跳空低开
            elif ctx.close[1] > ctx.ttin['min'][2] and ctx.open < ctx.ttin['min'][1] and \
                    self.num_unit == 0 and\
                    op_cycle\
                    and self.predict.signal(ctx.curbar) <= 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                if _unit > 0:
                    _open = copy.deepcopy(ctx.open)
                    self.breakpoint.append([_open, _unit, 'short'])
                    ctx.short(ctx.open, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')
            elif ctx.close[1] > ctx.ttin['min'][2] and ctx.low < ctx.ttin['min'][1] and \
                    self.num_unit == 0 and\
                    op_cycle\
                    and self.predict.signal(ctx.curbar) <= 0:
                self.n20 = self.N20
                _unit = self.risk_ctl.unitctl(ctx, self.n20)
                if _unit > 0:
                    _price = copy.deepcopy(ctx.ttin['min'][1])
                    self.breakpoint.append([_price, _unit, 'short'])
                    ctx.short(_price, _unit)
                    self.num_unit -= 1
                    self.display(ctx, 'short')

            # 空头退市
            elif ctx.close[1] < ctx.ttout['max'][2] and ctx.high > ctx.ttout['max'][1] and \
                    self.num_unit < 0 and\
                    op_cycle:
                all_units = 0
                for lst in self.breakpoint:
                    all_units += lst[1]
                    assert lst[2] == 'short'
                self.breakpoint = []
                assert isinstance(all_units, int)
                assert all_units > 0
                print ('cover_unit = %d' % all_units)
                _price = self._cmp(ctx.high, ctx.low, ctx.ttout['max'][1])
                ctx.cover(_price, all_units)
                self.num_unit = 0
                self.display(ctx, 'cover')

            # 多头加仓
            elif 0 < self.num_unit < maxunit:
                # 跳空高开
                if (self.breakpoint[-1][0] + self.n20 * 0.5) < ctx.open and\
                        op_cycle\
                        and self.predict.signal(ctx.curbar) >= 0:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint.append([_open, _unit, 'buy'])
                        ctx.buy(ctx.open, _unit)
                        self.num_unit += 1
                        self.display(ctx, 'buy')
                elif (self.breakpoint[-1][0] + self.n20 * 0.5) < ctx.high and\
                        op_cycle\
                        and self.predict.signal(ctx.curbar) >= 0:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    if _unit > 0:
                        _price = round(self.breakpoint[-1][0] + self.n20 * 0.5, 2)
                        self.breakpoint.append([_price, _unit, 'buy'])
                        ctx.buy(_price, _unit)
                        self.num_unit += 1
                        self.display(ctx, 'buy')
                # 多头止损
                elif (self.breakpoint[-1][0] - self.n20 * 2) > ctx.low and\
                        op_cycle:
                    all_units = 0
                    p = round(self.breakpoint[-1][0] - self.n20 * 2, 2)
                    _price = self._cmp(ctx.high, ctx.low, p)
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)

                    ctx.sell(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'sell')

            # 空头加仓
            elif _maxunit < self.num_unit < 0:
                # 跳空低开
                if (self.breakpoint[-1][0] - self.n20 * 0.5) > ctx.open and\
                        op_cycle\
                        and self.predict.signal(ctx.curbar) <= 0:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint.append([_open, _unit, 'short'])
                        ctx.short(ctx.open, _unit)
                        self.num_unit -= 1
                        self.display(ctx, 'short')
                elif (self.breakpoint[-1][0] - self.n20 * 0.5) > ctx.low and\
                        op_cycle\
                        and self.predict.signal(ctx.curbar) <= 0:
                    _unit = self.risk_ctl.unitctl(ctx, self.n20)
                    if _unit > 0:
                        _price = round(self.breakpoint[-1][0] - self.n20 * 0.5, 2)
                        self.breakpoint.append([_price, _unit, 'short'])
                        ctx.short(_price, _unit)
                        self.num_unit -= 1
                        self.display(ctx, 'short')
                # 空头止损
                elif (self.breakpoint[-1][0] + self.n20 * 2) < ctx.high and\
                        op_cycle:
                    all_units = 0
                    p = round(self.breakpoint[-1][0] + self.n20 * 2, 2)
                    _price = self._cmp(ctx.high, ctx.low, p)
                    for lst in self.breakpoint:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    self.breakpoint = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    ctx.cover(_price, all_units)
                    self.num_unit = 0
                    self.display(ctx, 'cover')

    def on_symbol(self, ctx):
        return

    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  Custom  **************')
        print ('N20 = %f, N = %f' % (self.n20, self.N[-1]))

        print('%s %s; Number of Units = %d' % (name, ctx.symbol, self.num_unit))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        print (ctx.position(tag, 'BB.SHFE'))

        """ ATTENTION: operation cycle reset signal !!! """
        if FILTER != 1:
            self.oc.counter(1)

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]


class RUN(object):
    def __init__(self):
        self.rst = [
            ['profile', '20 Days'],
            # ['profile2', '55 Days'],
            # ['profile3', '20 Days + Pred'],
            # ['profile4', 'Custom']
        ]
        self.pf = dict()

        set_symbols([CTS])
        self.pf['profile'] = add_strategy([DemoStrategy('20 Days')], {'capital': CAPITAL})
        # self.pf['profile2'] = add_strategy([DemoStrategy2('55 Days')], {'capital': CAPITAL})
        # self.pf['profile3'] = add_strategy([DemoStrategy3('20 Days + Predict')], {'capital': CAPITAL})
        # self.pf['profile4'] = add_strategy([MainStrategy('Custom')], {'capital': CAPITAL})

    def results(self):
        start = timeit.default_timer()
        run()
        stop = timeit.default_timer()
        print "运行耗时: %d秒" % (stop - start)

        filename = './Output/' + Default.get_name() + '.txt'
        f = open(filename, 'w')

        for pf in self.rst:
            curve = finance.create_equity_curve(self.pf[pf[0]].all_holdings())
            print pf[1], finance.summary_stats(curve, 252 * 4 * 60)

            f.write('\n')
            f.write(pf[1])
            f.write('\n')
            f.write(str(finance.summary_stats(curve, 252 * 4 * 60)))

            '''
            filename = './Output/' + Default.get_name() + '.txt'
            with open(filename, 'w') as f:
                f.write('\n')
                f.write(pf[1])
                f.write('\n')
                f.write(str(finance.summary_stats(curve, 252 * 4 * 60)))
            '''

            figname = Default.get_name() + pf[1]

            # K-line
            myplot.plot_strategy(self.pf[pf[0]].data(0), self.pf[pf[0]].technicals(0), self.pf[pf[0]].deals(0),
                                 curve.equity, figname)

            # Equity-curve
            # myplot.plot_curves([curve.equity], colors=['r'], names=[self.pf[pf[0]].name(0) + '-equity'])
            # myplot.plot_curves([curve.cash], colors=['g'], names=[self.pf[pf[0]].name(0) + '-cash'])
            # myplot.plot_curves([curve.returns], colors=['b'], names=[self.pf[pf[0]].name(0) + '-returns'])
            # myplot.plot_curves([curve.equity, curve.cash, curve.returns, curve.commission],
            #                    colors=['r', 'g', 'b', 'y'],
            #                    names=[self.pf[pf[0]].name(0)])

        f.close()

'''
if __name__ == '__main__':
    from quantdigger.digger import finance, plotting
    import myplot

    set_symbols([CTS])
    profile = add_strategy([DemoStrategy('20 Days')], {'capital': CAPITAL})
    # profile2 = add_strategy([DemoStrategy2('55 Days')], {'capital': 10000000})
    # profile3 = add_strategy([DemoStrategy3('20 Days + Predict')], {'capital': 10000000})
    profile4 = add_strategy([MainStrategy('Custom')], {'capital': CAPITAL})

    start = timeit.default_timer()
    run()
    stop = timeit.default_timer()
    print "运行耗时: %d秒" % (stop - start)

    # 打印组合1的统计信息
    curve = finance.create_equity_curve(profile.all_holdings())
    print '20 Days', finance.summary_stats(curve, 252 * 4 * 60)
    # curve2 = finance.create_equity_curve(profile2.all_holdings())
    # print '55 Days', finance.summary_stats(curve2, 252 * 4 * 60)
    # curve3 = finance.create_equity_curve(profile3.all_holdings())
    # print '20 Days + Predict', finance.summary_stats(curve3, 252 * 4 * 60)
    curve4 = finance.create_equity_curve(profile4.all_holdings())
    print 'Custom', finance.summary_stats(curve4, 252 * 4 * 60)

    filename = './Output/' + Default.get_name() + '.txt'
    with open(filename, 'w') as f:
        f.write('\n 20 Days \n')
        f.write(str(finance.summary_stats(curve, 252 * 4 * 60)))
        # f.write('\n 55 Days \n')
        # f.write(str(finance.summary_stats(curve2, 252 * 4 * 60)))
        # f.write('\n 20 Days + Predict \n')
        # f.write(str(finance.summary_stats(curve3, 252 * 4 * 60)))

    figname = Default.get_name() + '20'
    # figname2 = Default.get_name() + '55'
    # figname3 = Default.get_name() + '22+Pred'
    figname4 = Default.get_name() + 'Custom'

    # K-line
    myplot.plot_strategy(profile.data(0), profile.technicals(0), profile.deals(0), curve.equity, figname)
    # myplot.plot_strategy(profile2.data(0), profile2.technicals(0), profile2.deals(0), curve2.equity, figname2)
    # myplot.plot_strategy(profile3.data(0), profile3.technicals(0), profile3.deals(0), curve3.equity, figname3)
    myplot.plot_strategy(profile4.data(0), profile4.technicals(0), profile4.deals(0), curve4.equity, figname4)

    # Equity-curve
    myplot.plot_curves([curve.equity], colors=['r'], names=[profile.name(0)])
    myplot.plot_curves([curve4.equity], colors=['r'], names=[profile4.name(0)])
    # myplot.plot_curves([curve.equity, curve2.equity, curve3.equity], colors=['r', 'b', 'g'],
                       # names=[profile.name(0), profile2.name(0), profile3.name(0)])

    myplot.show()

'''

if __name__ == '__main__':
    from quantdigger.digger import finance, plotting
    import myplot
    import time

    Reference.check()
    time.sleep(5)
    P = RUN()
    P.results()
    myplot.show()



