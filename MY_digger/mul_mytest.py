# -*- coding: utf-8 -*-

from mul_base import *
from quantdigger import *
import timeit, copy
from mul_config import Default
from mul_config import PERIOD
from quantdigger.digger import finance, plotting
import myplot
import time
import datetime


CAPITAL = Default.get_capital()
RANDOM = Default.get_random()
MESSAGE = Messages(CAPITAL)

# history tt
class DemoStrategy(Strategy):

    def on_init(self, ctx):
        """初始化数据"""
        ctx.tt20 = Turtle_base(ctx.close, 20, 'tt20', ('r', 'g'))
        ctx.tt10 = Turtle_base(ctx.close, 10, 'tt10', ('y', 'b'))

        self.N_mul = dict()
        self.N20_mul = dict()
        self.n20_mul = dict()
        self.num_unit_mul = dict()
        # self.max_unit_mul = dict()
        self.breakpoint_mul = dict()

        self.risk_ctl = Risk_ctl(ctx)
        self.restraint = Restraint()

        self.L = []

        self.candicates = dict()

        print 'ctx.symbol = %s' % ctx.symbol.split('.')[0]

    def on_symbol(self, ctx):
        if ctx.curbar == 1:
            self.N_mul[ctx.symbol] = []
            self.N20_mul[ctx.symbol] = 0.0
            self.n20_mul[ctx.symbol] = 0.0
            self.num_unit_mul[ctx.symbol] = 0
            # self.max_unit_mul[ctx.symbol] = 4
            self.breakpoint_mul[ctx.symbol] = []

        if ctx.curbar > 1:
            self.L.append(max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low)))
            self.N_mul[ctx.symbol] = self.L

            if ctx.curbar > 20:
                for i in range(1, 20):
                    self.N20_mul[ctx.symbol] += self.N_mul[ctx.symbol][i]
                self.N20_mul[ctx.symbol] /= len(self.N_mul[ctx.symbol])

        if ctx.curbar > 20:
            self.N20_mul[ctx.symbol] = (self.N20_mul[ctx.symbol] * 19 + self.N_mul[ctx.symbol][-1]) / 20.0
            # print (ctx.symbol, self.N20_mul[ctx.symbol])

            # 多头入市
            # 跳空高开
            if ctx.close[1] < ctx.tt20['max'][2] and ctx.open > ctx.tt20['max'][1] and RANDOM and\
                    self.num_unit_mul[ctx.symbol] == 0 and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                    self.candicates[ctx.symbol] = 'buy 1a'

            elif ctx.close[1] < ctx.tt20['max'][2] and ctx.high > ctx.tt20['max'][1] and RANDOM and\
                    self.num_unit_mul[ctx.symbol] == 0 and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                    self.candicates[ctx.symbol] = 'buy 1b'

            # 多头退市
            elif ctx.close[1] > ctx.tt10['min'][2] and ctx.low < ctx.tt10['min'][1] and \
                    self.num_unit_mul[ctx.symbol] > 0:
                self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                self.candicates[ctx.symbol] = 'sell 1'

            # 空头入市
            # 跳空低开
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.open < ctx.tt20['min'][1] and RANDOM and\
                    self.num_unit_mul[ctx.symbol] == 0 and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                    self.candicates[ctx.symbol] = 'short 1a'
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.low < ctx.tt20['min'][1] and RANDOM and\
                    self.num_unit_mul[ctx.symbol] == 0 and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                    self.candicates[ctx.symbol] = 'short 1b'

                # 空头退市
            elif ctx.close[1] < ctx.tt10['max'][2] and ctx.high > ctx.tt10['max'][1] and \
                    self.num_unit_mul[ctx.symbol] < 0:
                self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                self.candicates[ctx.symbol] = 'cover 1'

                # 多头加仓
            elif 0 < self.num_unit_mul[ctx.symbol] < 4:
                # 跳空高开
                if (self.breakpoint_mul[ctx.symbol][-1][0] + self.n20_mul[ctx.symbol] * 0.5) < ctx.open \
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                        self.candicates[ctx.symbol] = 'buy 2a'
                elif (self.breakpoint_mul[ctx.symbol][-1][0] + self.n20_mul[ctx.symbol] * 0.5) < ctx.high\
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                        self.candicates[ctx.symbol] = 'buy 2b'
                    # 多头止损
                elif (self.breakpoint_mul[ctx.symbol][-1][0] - self.n20_mul[ctx.symbol] * 2) > ctx.low:
                    self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                    self.candicates[ctx.symbol] = 'sell 2'

                    # 空头加仓
            elif -4 < self.num_unit_mul[ctx.symbol] < 0:
                # 跳空低开
                if (self.breakpoint_mul[ctx.symbol][-1][0] - self.n20_mul[ctx.symbol] * 0.5) > ctx.open\
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                        self.candicates[ctx.symbol] = 'short 2a'
                elif (self.breakpoint_mul[ctx.symbol][-1][0] - self.n20_mul[ctx.symbol] * 0.5) > ctx.low\
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                        self.candicates[ctx.symbol] = 'short 2b'
                    # 空头止损
                elif (self.breakpoint_mul[ctx.symbol][-1][0] + self.n20_mul[ctx.symbol] * 2) < ctx.high:
                    self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                    self.candicates[ctx.symbol] = 'cover 2'

    def on_bar(self, ctx):
        if self.candicates:
            print self.candicates
            for cand in self.candicates:
                print('cand = %s' % cand)
                cash = ctx.cash()

                # force cover
                if cash < 0:
                    print ('### Current print = %f' % cash)
                    if self.num_unit_mul[cand] > 0:
                        all_units = 0
                        for lst in self.breakpoint_mul[cand]:
                            all_units += lst[1]
                            assert lst[2] == 'buy'
                        self.breakpoint_mul[cand] = []
                        assert isinstance(all_units, int)
                        assert all_units > 0
                        print ('sell_unit = %d' % all_units)
                        _price = self._cmp(ctx.high, ctx.low, ctx.tt10['min'][1])
                        ctx.sell(_price, all_units)
                        self.num_unit_mul[cand]  = 0
                        self.display(ctx, 'sell-force', cand)
                    elif self.num_unit_mul[cand]  < 0:
                        all_units = 0
                        for lst in self.breakpoint_mul[cand]:
                            all_units += lst[1]
                            assert lst[2] == 'short'
                        self.breakpoint_mul[cand] = []
                        assert isinstance(all_units, int)
                        assert all_units > 0
                        print ('cover_unit = %d' % all_units)
                        _price = self._cmp(ctx.high, ctx.low, ctx.tt10['max'][1])
                        ctx.cover(_price, all_units)
                        self.num_unit_mul[cand]  = 0
                        self.display(ctx, 'cover-force', cand)

                    continue

                # 空开跳高
                if self.candicates[cand] == 'buy 1a':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'buy'])
                        ctx.buy(ctx.open, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                        print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 多头开仓
                elif self.candicates[cand] == 'buy 1b':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = copy.deepcopy(ctx.tt20['max'][1])
                        self.breakpoint_mul[cand].append([_price, _unit, 'buy'])
                        ctx.buy(_price, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 多头退市
                elif self.candicates[cand] == 'sell 1':
                    all_units = 0
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)

                    _price = self._cmp(ctx.open, ctx.low, ctx.tt10['min'][1])
                    ctx.sell(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'sell', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

                # 空开跳低
                elif self.candicates[cand] == 'short 1a':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'short'])
                        ctx.short(ctx.open, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 空头开仓
                elif self.candicates[cand] == 'short 1b':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = copy.deepcopy(ctx.tt20['min'][1])
                        self.breakpoint_mul[cand].append([_price, _unit, 'short'])
                        ctx.short(_price, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 空头退市
                elif self.candicates[cand] == 'cover 1':
                    all_units = 0
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.open, ctx.tt10['max'][1])
                    ctx.cover(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'cover', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

                # 多头加仓 跳空高开
                elif self.candicates[cand] == 'buy 2a':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'buy'])
                        ctx.buy(ctx.open, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                #
                elif self.candicates[cand] == 'buy 2b':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = round(self.breakpoint_mul[cand][-1][0] + self.n20_mul[cand] * 0.5, 2)
                        self.breakpoint_mul[cand].append([_price, _unit, 'buy'])
                        ctx.buy(_price, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 多头止损
                elif self.candicates[cand] == 'sell 2':
                    all_units = 0
                    p = round(self.breakpoint_mul[cand][-1][0] - self.n20_mul[cand] * 2, 2)
                    _price = self._cmp(ctx.open, ctx.low, p)
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)

                    ctx.sell(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'sell', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

                # 空头加仓 跳空低开
                elif self.candicates[cand] == 'short 2a':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'short'])
                        ctx.short(ctx.open, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                #
                elif self.candicates[cand] == 'short 2b':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = round(self.breakpoint_mul[cand][-1][0] - self.n20_mul[cand] * 0.5, 2)
                        self.breakpoint_mul[cand].append([_price, _unit, 'short'])
                        ctx.short(_price, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 空头止损
                elif self.candicates[cand] == 'cover 2':
                    all_units = 0
                    p = round(self.breakpoint_mul[cand][-1][0] + self.n20_mul[cand] * 2, 2)
                    _price = self._cmp(ctx.high, ctx.open, p)
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    ctx.cover(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'cover', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

            self.candicates.clear()

    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless', cand='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  20 Days  **************')
        print ('N20 = %f, N = %f' % (self.n20_mul[cand], self.N_mul[cand][-1]))

        print('%s %s; Number of Units = %d' % (name, cand, self.num_unit_mul[cand]))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        # print (ctx.position(tag, 'BB.SHFE'))
        print ('datetime = %s' % str(ctx.datetime))

        MESSAGE.get_info(cash, name, cand=cand)

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]

# history strtt
class MainStrategy(Strategy):

    def on_init(self, ctx):
        """初始化数据"""
        self.tt = Default.get_tt()

        ctx.ttin = Turtle_base(ctx.close, self.tt['IN'], 'ttin', ('r', 'g'))
        ctx.ttout = Turtle_base(ctx.close, self.tt['OUT'], 'ttout', ('y', 'b'))

        self.num_unit_mul = dict()
        self.max_unit = Default.get_maxunit()
        self.N_mul = dict()
        self.N20_mul = dict()
        self.n20_mul = dict()

        self.oc = Operation_cycle()

        self.breakpoint_mul = dict()
        self.predict = Predict_module(CTS)

        self.restraint = Restraint()

        self.risk_ctl = Risk_ctl(ctx)

        self.L = []

        self.candicates = dict()

    def on_symbol(self, ctx):

        _maxunits = self.max_unit * -1
        maxunits = self.max_unit

        if ctx.curbar == 1:
            self.N_mul[ctx.symbol] = []
            self.N20_mul[ctx.symbol] = 0.0
            self.n20_mul[ctx.symbol] = 0.0
            self.num_unit_mul[ctx.symbol] = 0
            # self.max_unit_mul[ctx.symbol] = 4
            self.breakpoint_mul[ctx.symbol] = []



        if ctx.curbar > 1:
            self.L.append(max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low)))
            self.N_mul[ctx.symbol] = self.L

            if ctx.curbar > self.tt['N']:
                for i in range(1, 20):
                    self.N20_mul[ctx.symbol] += self.N_mul[ctx.symbol][i]
                self.N20_mul[ctx.symbol] /= len(self.N_mul[ctx.symbol])

        if ctx.curbar > self.tt['N']:
            self.N20_mul[ctx.symbol] = (self.N20_mul[ctx.symbol] * (self.tt['N'] - 1)
                                        + self.N_mul[ctx.symbol][-1]) / self.tt['N']
            # print (ctx.symbol, self.N20_mul[ctx.symbol])

            oc_cycle = self.oc.counter()

            # 多头入市
            # 跳空高开
            if ctx.close[1] < ctx.ttin['max'][2] and ctx.open > ctx.ttin['max'][1] and RANDOM and \
                    self.num_unit_mul[ctx.symbol] == 0 and oc_cycle and self.predict.signal(ctx.symbol, ctx.curbar) >= 0 \
                    and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                    self.candicates[ctx.symbol] = 'buy 1a'

            elif ctx.close[1] < ctx.ttin['max'][2] and ctx.high > ctx.ttin['max'][1] and RANDOM and \
                    self.num_unit_mul[ctx.symbol] == 0 and oc_cycle and self.predict.signal(ctx.symbol, ctx.curbar) >= 0\
                    and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                    self.candicates[ctx.symbol] = 'buy 1b'

            # 多头退市
            elif ctx.close[1] > ctx.ttout['min'][2] and ctx.low < ctx.ttout['min'][1] and \
                    self.num_unit_mul[ctx.symbol] > 0 and oc_cycle:
                self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                self.candicates[ctx.symbol] = 'sell 1'

            # 空头入市
            # 跳空低开
            elif ctx.close[1] > ctx.ttin['min'][2] and ctx.open < ctx.ttin['min'][1] and RANDOM and \
                    self.num_unit_mul[ctx.symbol] == 0 and oc_cycle and self.predict.signal(ctx.symbol, ctx.curbar) <= 0\
                    and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                    self.candicates[ctx.symbol] = 'short 1a'
            elif ctx.close[1] > ctx.ttin['min'][2] and ctx.low < ctx.ttin['min'][1] and RANDOM and \
                    self.num_unit_mul[ctx.symbol] == 0 and oc_cycle and self.predict.signal(ctx.symbol, ctx.curbar) <= 0\
                    and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                    self.candicates[ctx.symbol] = 'short 1b'

                # 空头退市
            elif ctx.close[1] < ctx.ttout['max'][2] and ctx.high > ctx.ttout['max'][1] and \
                    self.num_unit_mul[ctx.symbol] < 0 and oc_cycle:
                self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                self.candicates[ctx.symbol] = 'cover 1'

                # 多头加仓
            elif 0 < self.num_unit_mul[ctx.symbol] < maxunits:
                # 跳空高开
                if (self.breakpoint_mul[ctx.symbol][-1][0] + self.n20_mul[ctx.symbol] * 0.5) < ctx.open and \
                        oc_cycle and self.predict.signal(ctx.symbol, ctx.curbar) >= 0\
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                        self.candicates[ctx.symbol] = 'buy 2a'
                elif (self.breakpoint_mul[ctx.symbol][-1][0] + self.n20_mul[ctx.symbol] * 0.5) < ctx.high and \
                        oc_cycle and self.predict.signal(ctx.symbol, ctx.curbar) >= 0\
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                        self.candicates[ctx.symbol] = 'buy 2b'
                    # 多头止损
                elif (self.breakpoint_mul[ctx.symbol][-1][0] - self.n20_mul[ctx.symbol] * 2) > ctx.low and \
                        oc_cycle:
                    self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                    self.candicates[ctx.symbol] = 'sell 2'

                    # 空头加仓
            elif _maxunits < self.num_unit_mul[ctx.symbol] < 0:
                # 跳空低开
                if (self.breakpoint_mul[ctx.symbol][-1][0] - self.n20_mul[ctx.symbol] * 0.5) > ctx.open and \
                        oc_cycle and self.predict.signal(ctx.symbol, ctx.curbar) <= 0\
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                        self.candicates[ctx.symbol] = 'short 2a'
                elif (self.breakpoint_mul[ctx.symbol][-1][0] - self.n20_mul[ctx.symbol] * 0.5) > ctx.low and \
                        oc_cycle and self.predict.signal(ctx.symbol, ctx.curbar) <= 0\
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                        self.candicates[ctx.symbol] = 'short 2b'
                    # 空头止损
                elif (self.breakpoint_mul[ctx.symbol][-1][0] + self.n20_mul[ctx.symbol] * 2) < ctx.high and \
                        oc_cycle:
                    self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                    self.candicates[ctx.symbol] = 'cover 2'

    def on_bar(self, ctx):
        if self.candicates:
            print self.candicates
            for cand in self.candicates:
                print('cand = %s' % cand)
                cash = ctx.cash()

                # force cover
                if cash < 0:
                    print ('### Current print = %f' % cash)
                    if self.num_unit > 0:
                        all_units = 0
                        for lst in self.breakpoint_mul[cand]:
                            all_units += lst[1]
                            assert lst[2] == 'buy'
                        self.breakpoint_mul[cand] = []
                        assert isinstance(all_units, int)
                        assert all_units > 0
                        print ('sell_unit = %d' % all_units)
                        _price = self._cmp(ctx.high, ctx.low, ctx.ttout['min'][1])
                        ctx.sell(_price, all_units)
                        self.num_unit = 0
                        self.display(ctx, 'sell-force', cand)
                    elif self.num_unit < 0:
                        all_units = 0
                        for lst in self.breakpoint_mul[cand]:
                            all_units += lst[1]
                            assert lst[2] == 'short'
                        self.breakpoint_mul[cand] = []
                        assert isinstance(all_units, int)
                        assert all_units > 0
                        print ('cover_unit = %d' % all_units)
                        _price = self._cmp(ctx.high, ctx.low, ctx.ttout['max'][1])
                        ctx.cover(_price, all_units)
                        self.num_unit = 0
                        self.display(ctx, 'cover-force', cand)

                # 空开跳高
                if self.candicates[cand] == 'buy 1a':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'buy'])
                        ctx.buy(ctx.open, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                        print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 多头开仓
                elif self.candicates[cand] == 'buy 1b':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = copy.deepcopy(ctx.ttin['max'][1])
                        self.breakpoint_mul[cand].append([_price, _unit, 'buy'])
                        ctx.buy(_price, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 多头退市
                elif self.candicates[cand] == 'sell 1':
                    all_units = 0
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)
                    _price = self._cmp(ctx.open, ctx.low, ctx.ttout['min'][1])
                    ctx.sell(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'sell', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

                # 空开跳低
                elif self.candicates[cand] == 'short 1a':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'short'])
                        ctx.short(ctx.open, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 空头开仓
                elif self.candicates[cand] == 'short 1b':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = copy.deepcopy(ctx.ttin['min'][1])
                        self.breakpoint_mul[cand].append([_price, _unit, 'short'])
                        ctx.short(_price, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 空头退市
                elif self.candicates[cand] == 'cover 1':
                    all_units = 0
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.open, ctx.ttout['max'][1])
                    ctx.cover(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'cover', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

                # 多头加仓 跳空高开
                elif self.candicates[cand] == 'buy 2a':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'buy'])
                        ctx.buy(ctx.open, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                #
                elif self.candicates[cand] == 'buy 2b':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = round(self.breakpoint_mul[cand][-1][0] + self.n20_mul[cand] * 0.5, 2)
                        self.breakpoint_mul[cand].append([_price, _unit, 'buy'])
                        ctx.buy(_price, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 多头止损
                elif self.candicates[cand] == 'sell 2':
                    all_units = 0
                    p = round(self.breakpoint_mul[cand][-1][0] - self.n20_mul[cand] * 2, 2)
                    _price = self._cmp(ctx.open, ctx.low, p)
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)

                    ctx.sell(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'sell', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

                # 空头加仓 跳空低开
                elif self.candicates[cand] == 'short 2a':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'short'])
                        ctx.short(ctx.open, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                #
                elif self.candicates[cand] == 'short 2b':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = round(self.breakpoint_mul[cand][-1][0] - self.n20_mul[cand] * 0.5, 2)
                        self.breakpoint_mul[cand].append([_price, _unit, 'short'])
                        ctx.short(_price, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 空头止损
                elif self.candicates[cand] == 'cover 2':
                    all_units = 0
                    p = round(self.breakpoint_mul[cand][-1][0] + self.n20_mul[cand] * 2, 2)
                    _price = self._cmp(ctx.high, ctx.open, p)
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    ctx.cover(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'cover', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

            self.candicates.clear()

    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless', cand='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  20 Days  **************')
        print ('N20 = %f, N = %f' % (self.n20_mul[cand], self.N_mul[cand][-1]))

        print('%s %s; Number of Units = %d' % (name, cand, self.num_unit_mul[cand]))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        print (ctx.position(tag, 'BB.SHFE'))
        MESSAGE.get_info(cash, name, cand=cand)

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]


''' 5 Strategies
Turtle
Strength Turtle
Beforehand Turtle
ATR Turtle
ATR Beforehand Turtle
'''


# TODO for all : force cover
# done
class Turtle(Strategy):

    def on_init(self, ctx):
        """初始化数据"""
        '''
        d = {
            '20min': 20, '1h': 60, '1day': 365, '5days': 1825, '20days': 7300, '60days': 21900,
            '10min': 10, '30min': 30, '3h': 180, '2days': 730, '10days': 3650, '30days': 10950,
            '80min': 80, '4h': 240, '4day': 1460,            '80days': 292000, '240days': 87600
        }
        ctx.tt = {}
        for i in d:
            print i
            ctx.tt[i] = Turtle_base(ctx.close, d[i], i, ('r', 'g'))
        '''
        ctx.tt20 = Turtle_base(ctx.close, 20, 'tt20', ('r', 'g'))
        ctx.tt10 = Turtle_base(ctx.close, 10, 'tt10', ('y', 'b'))

        self.N_mul = dict()
        self.N20_mul = dict()
        self.n20_mul = dict()
        self.num_unit_mul = dict()
        self.breakpoint_mul = dict()

        self.risk_ctl = Risk_ctl(ctx)
        self.restraint = Restraint()

        self.L = []

        self.candicates = dict()

        print 'ctx.symbol = %s' % ctx.symbol.split('.')[0]      # eg. 'A1601'

        '''********************************'''

        self.count = Timecount(ctx)
        print self.count

        self.status = {}
        self.timecount = {}
        self.activepcon = Activepcon()
        self.freqcount = {}

    def on_symbol(self, ctx):
        """产生交易信号"""

        #self.count = Timecount(ctx)

        if ctx.curbar == 1:
            self.N_mul[ctx.symbol] = []
            self.N20_mul[ctx.symbol] = 0.0
            self.n20_mul[ctx.symbol] = 0.0
            self.num_unit_mul[ctx.symbol] = 0
            self.breakpoint_mul[ctx.symbol] = []

            '''ATR type: 1. 20min 2. 1h 3. 1day 4. 5days 5. 20days 6. 60days'''
            self.status[ctx.symbol] = Constatus()
            self.timecount[ctx.symbol] = Timecount(ctx)
            #self.activepcon = Activepcon()
            self.freqcount[ctx.symbol] = 0


        if ctx.curbar > 1:
            self.L.append(max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low)))
            self.N_mul[ctx.symbol] = self.L

            import time

            print ctx.datetime, ctx.open, ctx.symbol, ctx.volume

            isactive = self.timecount[ctx.symbol]._count(ctx)      #return if period active
            self.status[ctx.symbol]._ATR(ctx)

            lst = []
            if isactive:
                self.status[ctx.symbol]._ATR(period=isactive)

                lst = self.activepcon.refresh(ctx.symbol, self.status)

                if '1day' in isactive[0]:
                    for i in self.freqcount:
                        self.freqcount[i] = 0




            '''Strategy start!'''
            for period in ['20min', '1h', '1day', '5days', '20days', '60days']:
                if self.status[ctx.symbol].atrfour[period] and [ctx.symbol, period] in lst:
                    '''write strategy here'''
                    pcon = self.status[ctx.symbol]
                    self.restraint.countall(self.status)

                    #print '^^^^^^^^^^^^^^^^^^^'
                    #print ctx.close[1], pcon.maxmin[period][0], ctx.high
                    stopall = Stopall(self.freqcount[ctx.symbol])

                    stopbuy = Stopbuy(ctxdate=ctx[ctx.symbol].datetime, dateend=ctx.symbol) and\
                              self.restraint.ismaxnuit(ctx.symbol) and stopall


                    if pcon.unit[period] == 0:
                        # 多头入市
                        if ctx.close[1] < pcon.maxmin[period][0] and ctx.high >= pcon.maxmin[period][0] and\
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'buy 1'
                            pcon.candicates[period] = 'buy 1'
                            self.candicates[ctx.symbol] = True
                            assert pcon.candicates[period] == self.status[ctx.symbol].candicates[period]


                        # 空头入市
                        elif ctx.close[1] > pcon.maxmin[period][1] and ctx.low <= pcon.maxmin[period][1] and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'short 1'
                            pcon.candicates[period] = 'short 1'
                            self.candicates[ctx.symbol] = True
                            assert pcon.candicates[period] == self.status[ctx.symbol].candicates[period]

                    elif 0 < pcon.unit[period] < 4:
                        # 多头加仓
                        if ctx.high >= pcon.breakpoint[period][-1] + pcon.bpatrhalf[period] and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'buy 2'
                            pcon.candicates[period] = 'buy 2'
                            self.candicates[ctx.symbol] = True


                    elif -4 < pcon.unit[period] < 0:
                        # 空头加仓
                        if ctx.low <= pcon.breakpoint[period][-1] - pcon.bpatrhalf[period] and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'short 2'
                            pcon.candicates[period] = 'short 2'
                            self.candicates[ctx.symbol] = True


                    elif 0 < pcon.unit[period]:
                        # 多头退市
                        if ctx.close[1] > pcon.maxminhalf[period][1] >= ctx.low and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'sell 2'
                            pcon.candicates[period] = 'sell 2'
                            self.candicates[ctx.symbol] = True


                        # 多头止损
                        elif ctx.low <= pcon.breakpoint[period][-1] - pcon.bpatrhalf[period] and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'sell 1'
                            pcon.candicates[period] = 'sell 1'
                            self.candicates[ctx.symbol] = True


                    elif pcon.unit[period] < 0:
                        # 空头退市
                        if ctx.close[1] < pcon.maxminhalf[period][0] <= ctx.high and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'cover 2'
                            pcon.candicates[period] = 'cover 2'
                            self.candicates[ctx.symbol] = True


                        # 空头止损
                        elif ctx.high >= pcon.breakpoint[period][-1] + pcon.bpatrhalf[period] and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'cover 1'
                            pcon.candicates[period] = 'cover 1'
                            self.candicates[ctx.symbol] = True


            '''
            if ctx.curbar > 20:
                for i in range(1, 20):
                    self.N20_mul[ctx.symbol] += self.N_mul[ctx.symbol][i]
                self.N20_mul[ctx.symbol] /= len(self.N_mul[ctx.symbol])
            '''
        '''
        # if ctx.curbar > 20:
        if None:
            self.N20_mul[ctx.symbol] = (self.N20_mul[ctx.symbol] * 19 + self.N_mul[ctx.symbol][-1]) / 20.0

            # 多头入市
            # 跳空高开
            if ctx.close[1] < ctx.tt20['max'][2] and ctx.open > ctx.tt20['max'][1] and RANDOM and\
                    self.num_unit_mul[ctx.symbol] == 0 and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                    self.candicates[ctx.symbol] = 'buy 1a'

            elif ctx.close[1] < ctx.tt20['max'][2] and ctx.high > ctx.tt20['max'][1] and RANDOM and\
                    self.num_unit_mul[ctx.symbol] == 0 and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                    self.candicates[ctx.symbol] = 'buy 1b'

            # 多头退市
            elif ctx.close[1] > ctx.tt10['min'][2] and ctx.low < ctx.tt10['min'][1] and \
                    self.num_unit_mul[ctx.symbol] > 0:
                self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                self.candicates[ctx.symbol] = 'sell 1'

            # 空头入市
            # 跳空低开
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.open < ctx.tt20['min'][1] and RANDOM and\
                    self.num_unit_mul[ctx.symbol] == 0 and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                    self.candicates[ctx.symbol] = 'short 1a'
            elif ctx.close[1] > ctx.tt20['min'][2] and ctx.low < ctx.tt20['min'][1] and RANDOM and\
                    self.num_unit_mul[ctx.symbol] == 0 and Stopbuy(ctxdate=ctx[ctx.symbol].datetime):
                if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                    self.candicates[ctx.symbol] = 'short 1b'

                # 空头退市
            elif ctx.close[1] < ctx.tt10['max'][2] and ctx.high > ctx.tt10['max'][1] and \
                    self.num_unit_mul[ctx.symbol] < 0:
                self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                self.candicates[ctx.symbol] = 'cover 1'

                # 多头加仓
            elif 0 < self.num_unit_mul[ctx.symbol] < 4:
                # 跳空高开
                if (self.breakpoint_mul[ctx.symbol][-1][0] + self.n20_mul[ctx.symbol] * 0.5) < ctx.open \
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                        self.candicates[ctx.symbol] = 'buy 2a'
                elif (self.breakpoint_mul[ctx.symbol][-1][0] + self.n20_mul[ctx.symbol] * 0.5) < ctx.high\
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=1):
                        self.candicates[ctx.symbol] = 'buy 2b'
                    # 多头止损
                elif (self.breakpoint_mul[ctx.symbol][-1][0] - self.n20_mul[ctx.symbol] * 2) > ctx.low:
                    self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                    self.candicates[ctx.symbol] = 'sell 2'

                    # 空头加仓
            elif -4 < self.num_unit_mul[ctx.symbol] < 0:
                # 跳空低开
                if (self.breakpoint_mul[ctx.symbol][-1][0] - self.n20_mul[ctx.symbol] * 0.5) > ctx.open\
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                        self.candicates[ctx.symbol] = 'short 2a'
                elif (self.breakpoint_mul[ctx.symbol][-1][0] - self.n20_mul[ctx.symbol] * 0.5) > ctx.low\
                        and Stopbuy(ctxdate=ctx[ctx.symbol].datetime) and RANDOM:
                    if self.restraint.count(pcon=ctx.symbol.split('.')[0], num_unit_mul=self.num_unit_mul, trend=-1):
                        self.candicates[ctx.symbol] = 'short 2b'
                    # 空头止损
                elif (self.breakpoint_mul[ctx.symbol][-1][0] + self.n20_mul[ctx.symbol] * 2) < ctx.high:
                    self.restraint.count_clear(symbol=ctx.symbol, num_unit_mul=self.num_unit_mul)
                    self.candicates[ctx.symbol] = 'cover 2'
        '''

    def on_bar(self, ctx):
        """交易信号处理"""
        if self.candicates:
            print 'self.candicates = %s' % self.candicates

            for cand in self.candicates:
                pcon = self.status[cand]
                for period in ['20min', '1h', '1day', '5days', '20days', '60days']:
                    print('cand = %s' % cand)
                    # cash = ctx.cash()
                    print pcon.candicates

                    # force cover
                    '''
                    if cash < 0:
                        print ('### Current print = %f' % cash)
                        if self.num_unit_mul[cand] > 0:
                            all_units = 0
                            for lst in self.breakpoint_mul[cand]:
                                all_units += lst[1]
                                assert lst[2] == 'buy'
                            self.breakpoint_mul[cand] = []
                            assert isinstance(all_units, int)
                            assert all_units > 0
                            print ('sell_unit = %d' % all_units)
                            _price = self._cmp(ctx.high, ctx.low, ctx.tt10['min'][1])
                            ctx.sell(_price, all_units)
                            self.num_unit_mul[cand]  = 0
                            self.display(ctx, 'sell-force', cand)
                        elif self.num_unit_mul[cand] < 0:
                            all_units = 0
                            for lst in self.breakpoint_mul[cand]:
                                all_units += lst[1]
                                assert lst[2] == 'short'
                            self.breakpoint_mul[cand] = []
                            assert isinstance(all_units, int)
                            assert all_units > 0
                            print ('cover_unit = %d' % all_units)
                            _price = self._cmp(ctx.high, ctx.low, ctx.tt10['max'][1])
                            ctx.cover(_price, all_units)
                            self.num_unit_mul[cand]  = 0
                            self.display(ctx, 'cover-force', cand)

                        continue
                    '''

                    # 多头开仓
                    if pcon.candicates[period] == 'buy 1':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            _price = self._cmp(ctx.open, ctx.high, pcon.maxmin[period][0])
                            ctx.buy(_price, _unit)
                            self.display(ctx, 'buy', cand)

                            pcon.unit[period] = 1
                            pcon.breakpoint[period].append(pcon.maxmin[period][0])
                            pcon.bpatrhalf[period] = pcon.atrhalf[period]
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1


                    # 多头退市
                    if pcon.candicates[period] == 'sell 2':
                        _unit = sum(pcon._unit[period])
                        _price = self._cmp(ctx.open, ctx.low, pcon.maxmin[period][1])
                        ctx.sell(_price, _unit)
                        self.display(ctx, 'sell', cand)

                        pcon.unit[period] = 0
                        pcon.breakpoint[period] = []
                        pcon.bpatrhalf[period] = 0
                        pcon._unit[period] = []

                        self.freqcount[cand] += 1

                    # 空头开仓
                    if pcon.candicates[period] == 'short 1':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            _price = self._cmp(ctx.open, ctx.low, pcon.maxmin[period][1])
                            ctx.short(_price, _unit)
                            self.display(ctx, 'short', cand)

                            pcon.unit[period] = 1
                            pcon.breakpoint[period].append(pcon.maxmin[period][1])
                            pcon.bpatrhalf[period] = pcon.atrhalf[period]
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1

                    # 空头退市
                    if pcon.candicates[period] == 'cover 2':
                        _unit = sum(pcon._unit[period])
                        _price = self._cmp(ctx.open, ctx.high, pcon.maxmin[period][0])
                        ctx.cover(_price, _unit)
                        self.display(ctx, 'cover', cand)

                        pcon.unit[period] = 0
                        pcon.breakpoint[period] = []
                        pcon.bpatrhalf[period] = 0
                        pcon._unit[period] = []

                        self.freqcount[cand] += 1


                    # 多头加仓
                    if pcon.candicates[period] == 'buy 2':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            p = pcon.breakpoint[period][-1] + pcon.bpatrhalf[period]
                            _price = self._cmp(ctx.open, ctx.high, p)
                            ctx.buy(_price, _unit)
                            self.display(ctx, 'buy', cand)

                            pcon.unit[period] += 1
                            pcon.breakpoint[period].append(pcon.breakpoint[period][-1] + pcon.bpatrhalf[period])
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1


                    # 空头加仓
                    if pcon.candicates[period] == 'short 2':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            p = pcon.breakpoint[period][-1] - pcon.bpatrhalf[period]
                            _price = self._cmp(ctx.open, ctx.low, p)
                            ctx.short(_price, _unit)
                            self.display(ctx, 'short', cand)

                            pcon.unit[period] -= 1
                            pcon.breakpoint[period].append(pcon.breakpoint[period][-1] - pcon.bpatrhalf[period])
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1



                    # 多头止损
                    if pcon.candicates[period] == 'sell 1':
                        _unit = pcon._unit[period][-1]
                        p = pcon.breakpoint[period][-1] - pcon.bpatrhalf[period]
                        _price = self._cmp(ctx.open, ctx.low, p)
                        ctx.sell(_price, _unit)
                        self.display(ctx, 'sell', cand)

                        pcon.unit[period] -= 1
                        pcon.breakpoint[period] = pcon.breakpoint[period][:-1]
                        pcon._unit[period] = pcon._unit[period][:-1]

                        self.freqcount[cand] += 1

                    # 空头止损
                    if pcon.candicates[period] == 'cover 1':
                        _unit = pcon._unit[period][-1]
                        p = pcon.breakpoint[period][-1] + pcon.bpatrhalf[period]
                        _price = self._cmp(ctx.open, ctx.high, p)
                        ctx.cover(_price, _unit)
                        self.display(ctx, 'cover', cand)

                        pcon.unit[period] += 1
                        pcon.breakpoint[period] = pcon.breakpoint[period][:-1]
                        pcon._unit[period] = pcon._unit[period][:-1]

                        self.freqcount[cand] += 1

                pcon.candicates = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}

            self.candicates.clear()


            '''
                # 空开跳高
                if self.candicates[cand] == 'buy 1a':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'buy'])
                        ctx.buy(ctx.open, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                        print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 多头开仓
                elif self.candicates[cand] == 'buy 1b':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = copy.deepcopy(ctx.tt20['max'][1])
                        self.breakpoint_mul[cand].append([_price, _unit, 'buy'])
                        ctx.buy(_price, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 多头退市
                elif self.candicates[cand] == 'sell 1':
                    all_units = 0
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)

                    _price = self._cmp(ctx.open, ctx.low, ctx.tt10['min'][1])
                    ctx.sell(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'sell', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

                # 空开跳低
                elif self.candicates[cand] == 'short 1a':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'short'])
                        ctx.short(ctx.open, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 空头开仓
                elif self.candicates[cand] == 'short 1b':
                    self.n20_mul[cand] = self.N20_mul[cand]
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = copy.deepcopy(ctx.tt20['min'][1])
                        self.breakpoint_mul[cand].append([_price, _unit, 'short'])
                        ctx.short(_price, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 空头退市
                elif self.candicates[cand] == 'cover 1':
                    all_units = 0
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    _price = self._cmp(ctx.high, ctx.open, ctx.tt10['max'][1])
                    ctx.cover(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'cover', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

                # 多头加仓 跳空高开
                elif self.candicates[cand] == 'buy 2a':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'buy'])
                        ctx.buy(ctx.open, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                #
                elif self.candicates[cand] == 'buy 2b':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = round(self.breakpoint_mul[cand][-1][0] + self.n20_mul[cand] * 0.5, 2)
                        self.breakpoint_mul[cand].append([_price, _unit, 'buy'])
                        ctx.buy(_price, _unit)
                        self.num_unit_mul[cand] += 1
                        self.display(ctx, 'buy', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 多头止损
                elif self.candicates[cand] == 'sell 2':
                    all_units = 0
                    p = round(self.breakpoint_mul[cand][-1][0] - self.n20_mul[cand] * 2, 2)
                    _price = self._cmp(ctx.open, ctx.low, p)
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'buy'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('sell_unit = %d' % all_units)

                    ctx.sell(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'sell', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))

                # 空头加仓 跳空低开
                elif self.candicates[cand] == 'short 2a':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _open = copy.deepcopy(ctx.open)
                        self.breakpoint_mul[cand].append([_open, _unit, 'short'])
                        ctx.short(ctx.open, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                #
                elif self.candicates[cand] == 'short 2b':
                    _unit = self.risk_ctl.unitctl(ctx, self.n20_mul[cand])
                    if _unit > 0:
                        _price = round(self.breakpoint_mul[cand][-1][0] - self.n20_mul[cand] * 0.5, 2)
                        self.breakpoint_mul[cand].append([_price, _unit, 'short'])
                        ctx.short(_price, _unit)
                        self.num_unit_mul[cand] -= 1
                        self.display(ctx, 'short', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], _unit))

                # 空头止损
                elif self.candicates[cand] == 'cover 2':
                    all_units = 0
                    p = round(self.breakpoint_mul[cand][-1][0] + self.n20_mul[cand] * 2, 2)
                    _price = self._cmp(ctx.high, ctx.open, p)
                    for lst in self.breakpoint_mul[cand]:
                        all_units += lst[1]
                        assert lst[2] == 'short'
                    print('all units = %d' % all_units)
                    self.breakpoint_mul[cand] = []
                    assert isinstance(all_units, int)
                    assert all_units > 0
                    print ('cover_unit = %d' % all_units)
                    ctx.cover(_price, all_units)
                    self.num_unit_mul[cand] = 0
                    self.display(ctx, 'cover', cand)
                    print('策略%s, 执行%s %d units' % (ctx.strategy, self.candicates[cand], all_units))
            '''


    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless', cand='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  20 Days  **************')
        print ('N20 = %f, N = %f' % (self.n20_mul[cand], self.N_mul[cand][-1]))

        print('%s %s; Number of Units = %d' % (name, cand, self.num_unit_mul[cand]))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        # print (ctx.position(tag, 'BB.SHFE'))
        print ('datetime = %s' % str(ctx.datetime))

        MESSAGE.get_info(cash, name, cand=cand)

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]

# done
class Strengthtt(Strategy):
    def on_init(self, ctx):
        """初始化数据"""
        '''
        d = {
            '20min': 20, '1h': 60, '1day': 365, '5days': 1825, '20days': 7300, '60days': 21900,
            '10min': 10, '30min': 30, '3h': 180, '2days': 730, '10days': 3650, '30days': 10950,
            '80min': 80, '4h': 240, '4day': 1460,            '80days': 292000, '240days': 87600
        }
        ctx.tt = {}
        for i in d:
            print i
            ctx.tt[i] = Turtle_base(ctx.close, d[i], i, ('r', 'g'))
        '''
        ctx.tt20 = Turtle_base(ctx.close, 20, 'tt20', ('r', 'g'))
        ctx.tt10 = Turtle_base(ctx.close, 10, 'tt10', ('y', 'b'))

        self.N_mul = dict()
        self.N20_mul = dict()
        self.n20_mul = dict()
        self.num_unit_mul = dict()
        self.breakpoint_mul = dict()

        self.risk_ctl = Risk_ctl(ctx)
        self.restraint = Restraint()

        self.L = []

        self.candicates = dict()

        print 'ctx.symbol = %s' % ctx.symbol.split('.')[0]      # eg. 'A1601'

        '''********************************'''

        self.count = Timecount(ctx)
        print self.count

        self.status = {}
        self.timecount = {}
        self.activepcon = Activepcon()
        self.freqcount = {}

        self.predict = Predict_module(CTS)

    def on_symbol(self, ctx):

        #self.count = Timecount(ctx)

        if ctx.curbar == 1:
            self.N_mul[ctx.symbol] = []
            self.N20_mul[ctx.symbol] = 0.0
            self.n20_mul[ctx.symbol] = 0.0
            self.num_unit_mul[ctx.symbol] = 0
            self.breakpoint_mul[ctx.symbol] = []

            '''ATR type: 1. 20min 2. 1h 3. 1day 4. 5days 5. 20days 6. 60days'''
            self.status[ctx.symbol] = Constatus()
            self.timecount[ctx.symbol] = Timecount(ctx)
            #self.activepcon = Activepcon()
            self.freqcount[ctx.symbol] = 0


        if ctx.curbar > 1:
            self.L.append(max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low)))
            self.N_mul[ctx.symbol] = self.L

            import time

            print ctx.datetime, ctx.open, ctx.symbol, ctx.volume

            isactive = self.timecount[ctx.symbol]._count(ctx)      #return if period active
            self.status[ctx.symbol]._ATR(ctx)

            lst = []
            if isactive:
                self.status[ctx.symbol]._ATR(period=isactive)

                lst = self.activepcon.refresh(ctx.symbol, self.status)

                if '1day' in isactive[0]:
                    for i in self.freqcount:
                        self.freqcount[i] = 0




            '''Strategy start!'''
            for period in ['20min', '1h', '1day', '5days', '20days', '60days']:
                if self.status[ctx.symbol].atrfour[period] and [ctx.symbol, period] in lst:
                    '''write strategy here'''
                    pcon = self.status[ctx.symbol]
                    self.restraint.countall(self.status)

                    #print '^^^^^^^^^^^^^^^^^^^'
                    #print ctx.close[1], pcon.maxmin[period][0], ctx.high
                    stopall = Stopall(self.freqcount[ctx.symbol])

                    stopbuy = Stopbuy(ctxdate=ctx[ctx.symbol].datetime, dateend=ctx.symbol) and\
                              self.restraint.ismaxnuit(ctx.symbol) and stopall


                    if pcon.unit[period] == 0:
                        # 多头入市
                        if ctx.close[1] < pcon.maxmin[period][0] and ctx.high >= pcon.maxmin[period][0] and\
                                not stopbuy and self.predict.signal(ctx.symbol, ctx.curbar):
                            print '-----------------------'
                            print ctx.symbol, period, 'buy 1'
                            pcon.candicates[period] = 'buy 1'
                            self.candicates[ctx.symbol] = True
                            assert pcon.candicates[period] == self.status[ctx.symbol].candicates[period]


                        # 空头入市
                        elif ctx.close[1] > pcon.maxmin[period][1] and ctx.low <= pcon.maxmin[period][1] and \
                                not stopbuy and self.predict.signal(ctx.symbol, ctx.curbar):
                            print '-----------------------'
                            print ctx.symbol, period, 'short 1'
                            pcon.candicates[period] = 'short 1'
                            self.candicates[ctx.symbol] = True
                            assert pcon.candicates[period] == self.status[ctx.symbol].candicates[period]

                    elif 0 < pcon.unit[period] < 4:
                        # 多头加仓
                        if ctx.high >= pcon.breakpoint[period][-1] + pcon.bpatrhalf[period] and \
                                not stopbuy and self.predict.signal(ctx.symbol, ctx.curbar):
                            print '-----------------------'
                            print ctx.symbol, period, 'buy 2'
                            pcon.candicates[period] = 'buy 2'
                            self.candicates[ctx.symbol] = True


                    elif -4 < pcon.unit[period] < 0:
                        # 空头加仓
                        if ctx.low <= pcon.breakpoint[period][-1] - pcon.bpatrhalf[period] and \
                                not stopbuy and self.predict.signal(ctx.symbol, ctx.curbar):
                            print '-----------------------'
                            print ctx.symbol, period, 'short 2'
                            pcon.candicates[period] = 'short 2'
                            self.candicates[ctx.symbol] = True


                    elif 0 < pcon.unit[period]:
                        # 多头退市
                        if (ctx.close[1] > pcon.maxminhalf[period][1] >= ctx.low or
                                Timefour(pcon.atrfour_count[period][0]) or BackStrengh(pcon.atrhalf[period])) and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'sell 2'
                            pcon.candicates[period] = 'sell 2'
                            self.candicates[ctx.symbol] = True


                        # 多头止损
                        elif ctx.low <= pcon.breakpoint[period][-1] - pcon.bpatrhalf[period] and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'sell 1'
                            pcon.candicates[period] = 'sell 1'
                            self.candicates[ctx.symbol] = True


                    elif pcon.unit[period] < 0:
                        # 空头退市
                        if (ctx.close[1] < pcon.maxminhalf[period][0] <= ctx.high or
                                Timefour(pcon.atrfour_count[period][0]) or BackStrengh(pcon.atrhalf[period])) and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'cover 2'
                            pcon.candicates[period] = 'cover 2'
                            self.candicates[ctx.symbol] = True


                        # 空头止损
                        elif ctx.high >= pcon.breakpoint[period][-1] + pcon.bpatrhalf[period] and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'cover 1'
                            pcon.candicates[period] = 'cover 1'
                            self.candicates[ctx.symbol] = True


    def on_bar(self, ctx):
        if self.candicates:
            print 'self.candicates = %s' % self.candicates

            for cand in self.candicates:
                pcon = self.status[cand]
                for period in ['20min', '1h', '1day', '5days', '20days', '60days']:
                    print('cand = %s' % cand)
                    # cash = ctx.cash()
                    print pcon.candicates

                    # force cover
                    '''
                    if cash < 0:
                        print ('### Current print = %f' % cash)
                        if self.num_unit_mul[cand] > 0:
                            all_units = 0
                            for lst in self.breakpoint_mul[cand]:
                                all_units += lst[1]
                                assert lst[2] == 'buy'
                            self.breakpoint_mul[cand] = []
                            assert isinstance(all_units, int)
                            assert all_units > 0
                            print ('sell_unit = %d' % all_units)
                            _price = self._cmp(ctx.high, ctx.low, ctx.tt10['min'][1])
                            ctx.sell(_price, all_units)
                            self.num_unit_mul[cand]  = 0
                            self.display(ctx, 'sell-force', cand)
                        elif self.num_unit_mul[cand] < 0:
                            all_units = 0
                            for lst in self.breakpoint_mul[cand]:
                                all_units += lst[1]
                                assert lst[2] == 'short'
                            self.breakpoint_mul[cand] = []
                            assert isinstance(all_units, int)
                            assert all_units > 0
                            print ('cover_unit = %d' % all_units)
                            _price = self._cmp(ctx.high, ctx.low, ctx.tt10['max'][1])
                            ctx.cover(_price, all_units)
                            self.num_unit_mul[cand]  = 0
                            self.display(ctx, 'cover-force', cand)

                        continue
                    '''

                    # 多头开仓
                    if pcon.candicates[period] == 'buy 1':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            _price = self._cmp(ctx.open, ctx.high, pcon.maxmin[period][0])
                            ctx.buy(_price, _unit)
                            self.display(ctx, 'buy', cand)

                            pcon.unit[period] = 1
                            pcon.breakpoint[period].append(pcon.maxmin[period][0])
                            pcon.bpatrhalf[period] = pcon.atrhalf[period]
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1


                    # 多头退市
                    if pcon.candicates[period] == 'sell 2':
                        _unit = sum(pcon._unit[period])
                        _price = self._cmp(ctx.open, ctx.low, pcon.maxmin[period][1])
                        ctx.sell(_price, _unit)
                        self.display(ctx, 'sell', cand)

                        pcon.unit[period] = 0
                        pcon.breakpoint[period] = []
                        pcon.bpatrhalf[period] = 0
                        pcon._unit[period] = []

                        self.freqcount[cand] += 1

                    # 空头开仓
                    if pcon.candicates[period] == 'short 1':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            _price = self._cmp(ctx.open, ctx.low, pcon.maxmin[period][1])
                            ctx.short(_price, _unit)
                            self.display(ctx, 'short', cand)

                            pcon.unit[period] = 1
                            pcon.breakpoint[period].append(pcon.maxmin[period][1])
                            pcon.bpatrhalf[period] = pcon.atrhalf[period]
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1

                    # 空头退市
                    if pcon.candicates[period] == 'cover 2':
                        _unit = sum(pcon._unit[period])
                        _price = self._cmp(ctx.open, ctx.high, pcon.maxmin[period][0])
                        ctx.cover(_price, _unit)
                        self.display(ctx, 'cover', cand)

                        pcon.unit[period] = 0
                        pcon.breakpoint[period] = []
                        pcon.bpatrhalf[period] = 0
                        pcon._unit[period] = []

                        self.freqcount[cand] += 1


                    # 多头加仓
                    if pcon.candicates[period] == 'buy 2':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            p = pcon.breakpoint[period][-1] + pcon.bpatrhalf[period]
                            _price = self._cmp(ctx.open, ctx.high, p)
                            ctx.buy(_price, _unit)
                            self.display(ctx, 'buy', cand)

                            pcon.unit[period] += 1
                            pcon.breakpoint[period].append(pcon.breakpoint[period][-1] + pcon.bpatrhalf[period])
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1


                    # 空头加仓
                    if pcon.candicates[period] == 'short 2':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            p = pcon.breakpoint[period][-1] - pcon.bpatrhalf[period]
                            _price = self._cmp(ctx.open, ctx.low, p)
                            ctx.short(_price, _unit)
                            self.display(ctx, 'short', cand)

                            pcon.unit[period] -= 1
                            pcon.breakpoint[period].append(pcon.breakpoint[period][-1] - pcon.bpatrhalf[period])
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1



                    # 多头止损
                    if pcon.candicates[period] == 'sell 1':
                        _unit = pcon._unit[period][-1]
                        p = pcon.breakpoint[period][-1] - pcon.bpatrhalf[period]
                        _price = self._cmp(ctx.open, ctx.low, p)
                        ctx.sell(_price, _unit)
                        self.display(ctx, 'sell', cand)

                        pcon.unit[period] -= 1
                        pcon.breakpoint[period] = pcon.breakpoint[period][:-1]
                        pcon._unit[period] = pcon._unit[period][:-1]

                        self.freqcount[cand] += 1

                    # 空头止损
                    if pcon.candicates[period] == 'cover 1':
                        _unit = pcon._unit[period][-1]
                        p = pcon.breakpoint[period][-1] + pcon.bpatrhalf[period]
                        _price = self._cmp(ctx.open, ctx.high, p)
                        ctx.cover(_price, _unit)
                        self.display(ctx, 'cover', cand)

                        pcon.unit[period] += 1
                        pcon.breakpoint[period] = pcon.breakpoint[period][:-1]
                        pcon._unit[period] = pcon._unit[period][:-1]

                        self.freqcount[cand] += 1

                pcon.candicates = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}

            self.candicates.clear()




    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless', cand='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  20 Days  **************')
        print ('N20 = %f, N = %f' % (self.n20_mul[cand], self.N_mul[cand][-1]))

        print('%s %s; Number of Units = %d' % (name, cand, self.num_unit_mul[cand]))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        # print (ctx.position(tag, 'BB.SHFE'))
        print ('datetime = %s' % str(ctx.datetime))

        MESSAGE.get_info(cash, name, cand=cand)

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]

# TODO predict.price
class Beforehandtt(Strategy):
    def on_init(self, ctx):
        """初始化数据"""
        '''
        d = {
            '20min': 20, '1h': 60, '1day': 365, '5days': 1825, '20days': 7300, '60days': 21900,
            '10min': 10, '30min': 30, '3h': 180, '2days': 730, '10days': 3650, '30days': 10950,
            '80min': 80, '4h': 240, '4day': 1460,            '80days': 292000, '240days': 87600
        }
        ctx.tt = {}
        for i in d:
            print i
            ctx.tt[i] = Turtle_base(ctx.close, d[i], i, ('r', 'g'))
        '''
        ctx.tt20 = Turtle_base(ctx.close, 20, 'tt20', ('r', 'g'))
        ctx.tt10 = Turtle_base(ctx.close, 10, 'tt10', ('y', 'b'))

        self.N_mul = dict()
        self.N20_mul = dict()
        self.n20_mul = dict()
        self.num_unit_mul = dict()
        self.breakpoint_mul = dict()

        self.risk_ctl = Risk_ctl(ctx)
        self.restraint = Restraint()

        self.L = []

        self.candicates = dict()

        print 'ctx.symbol = %s' % ctx.symbol.split('.')[0]      # eg. 'A1601'

        '''********************************'''

        self.count = Timecount(ctx)
        print self.count

        self.status = {}
        self.timecount = {}
        self.activepcon = Activepcon()
        self.freqcount = {}

        self.predict = Predict_module(CTS)

    def on_symbol(self, ctx):

        #self.count = Timecount(ctx)

        if ctx.curbar == 1:
            self.N_mul[ctx.symbol] = []
            self.N20_mul[ctx.symbol] = 0.0
            self.n20_mul[ctx.symbol] = 0.0
            self.num_unit_mul[ctx.symbol] = 0
            self.breakpoint_mul[ctx.symbol] = []

            '''ATR type: 1. 20min 2. 1h 3. 1day 4. 5days 5. 20days 6. 60days'''
            self.status[ctx.symbol] = Constatus()
            self.timecount[ctx.symbol] = Timecount(ctx)
            #self.activepcon = Activepcon()
            self.freqcount[ctx.symbol] = 0


        if ctx.curbar > 1:
            self.L.append(max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low)))
            self.N_mul[ctx.symbol] = self.L

            import time

            print ctx.datetime, ctx.open, ctx.symbol, ctx.volume

            isactive = self.timecount[ctx.symbol]._count(ctx)      #return if period active
            self.status[ctx.symbol]._ATR(ctx)

            lst = []
            if isactive:
                self.status[ctx.symbol]._ATR(period=isactive)

                lst = self.activepcon.refresh(ctx.symbol, self.status)

                if '1day' in isactive[0]:
                    for i in self.freqcount:
                        self.freqcount[i] = 0




            '''Strategy start!'''
            for period in ['20min', '1h', '1day', '5days', '20days', '60days']:
                if self.status[ctx.symbol].atrfour[period] and [ctx.symbol, period] in lst:
                    '''write strategy here'''
                    pcon = self.status[ctx.symbol]
                    self.restraint.countall(self.status)

                    #print '^^^^^^^^^^^^^^^^^^^'
                    #print ctx.close[1], pcon.maxmin[period][0], ctx.high
                    stopall = Stopall(self.freqcount[ctx.symbol])

                    stopbuy = Stopbuy(ctxdate=ctx[ctx.symbol].datetime, dateend=ctx.symbol) and\
                              self.restraint.ismaxnuit(ctx.symbol) and stopall


                    if pcon.unit[period] == 0:
                        # 多头入市
                        if ctx.close[1] < pcon.maxmin[period][0] and \
                                        self.predict.price(ctx.symbol, ctx.curbar) >= pcon.maxmin[period][0] and\
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'buy 1'
                            pcon.candicates[period] = 'buy 1'
                            self.candicates[ctx.symbol] = True
                            assert pcon.candicates[period] == self.status[ctx.symbol].candicates[period]


                        # 空头入市
                        elif ctx.close[1] > pcon.maxmin[period][1] and \
                                        self.predict.price(ctx.symbol, ctx.curbar) <= pcon.maxmin[period][1] and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'short 1'
                            pcon.candicates[period] = 'short 1'
                            self.candicates[ctx.symbol] = True
                            assert pcon.candicates[period] == self.status[ctx.symbol].candicates[period]

                    elif 0 < pcon.unit[period] < 4:
                        # 多头加仓
                        if self.predict.price(ctx.symbol, ctx.curbar) >= pcon.breakpoint[period][-1] + pcon.bpatrhalf[period] and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'buy 2'
                            pcon.candicates[period] = 'buy 2'
                            self.candicates[ctx.symbol] = True


                    elif -4 < pcon.unit[period] < 0:
                        # 空头加仓
                        if self.predict.price(ctx.symbol, ctx.curbar)<= pcon.breakpoint[period][-1] - pcon.bpatrhalf[period] and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'short 2'
                            pcon.candicates[period] = 'short 2'
                            self.candicates[ctx.symbol] = True


                    elif 0 < pcon.unit[period]:
                        # 多头退市
                        if (ctx.close[1] > pcon.maxminhalf[period][1] >= ctx.low or
                                Timefour(pcon.atrfour_count[period][0]) or BackStrengh(pcon.atrhalf[period])) and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'sell 2'
                            pcon.candicates[period] = 'sell 2'
                            self.candicates[ctx.symbol] = True


                        # 多头止损
                        elif ctx.low <= pcon.breakpoint[period][-1] - pcon.bpatrhalf[period] and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'sell 1'
                            pcon.candicates[period] = 'sell 1'
                            self.candicates[ctx.symbol] = True


                    elif pcon.unit[period] < 0:
                        # 空头退市
                        if (ctx.close[1] < pcon.maxminhalf[period][0] <= ctx.high or
                                Timefour(pcon.atrfour_count[period][0]) or BackStrengh(pcon.atrhalf[period])) and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'cover 2'
                            pcon.candicates[period] = 'cover 2'
                            self.candicates[ctx.symbol] = True


                        # 空头止损
                        elif ctx.high >= pcon.breakpoint[period][-1] + pcon.bpatrhalf[period] and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'cover 1'
                            pcon.candicates[period] = 'cover 1'
                            self.candicates[ctx.symbol] = True


    def on_bar(self, ctx):
        if self.candicates:
            print 'self.candicates = %s' % self.candicates

            for cand in self.candicates:
                pcon = self.status[cand]
                for period in ['20min', '1h', '1day', '5days', '20days', '60days']:
                    print('cand = %s' % cand)
                    # cash = ctx.cash()
                    print pcon.candicates

                    # force cover
                    '''
                    if cash < 0:
                        print ('### Current print = %f' % cash)
                        if self.num_unit_mul[cand] > 0:
                            all_units = 0
                            for lst in self.breakpoint_mul[cand]:
                                all_units += lst[1]
                                assert lst[2] == 'buy'
                            self.breakpoint_mul[cand] = []
                            assert isinstance(all_units, int)
                            assert all_units > 0
                            print ('sell_unit = %d' % all_units)
                            _price = self._cmp(ctx.high, ctx.low, ctx.tt10['min'][1])
                            ctx.sell(_price, all_units)
                            self.num_unit_mul[cand]  = 0
                            self.display(ctx, 'sell-force', cand)
                        elif self.num_unit_mul[cand] < 0:
                            all_units = 0
                            for lst in self.breakpoint_mul[cand]:
                                all_units += lst[1]
                                assert lst[2] == 'short'
                            self.breakpoint_mul[cand] = []
                            assert isinstance(all_units, int)
                            assert all_units > 0
                            print ('cover_unit = %d' % all_units)
                            _price = self._cmp(ctx.high, ctx.low, ctx.tt10['max'][1])
                            ctx.cover(_price, all_units)
                            self.num_unit_mul[cand]  = 0
                            self.display(ctx, 'cover-force', cand)

                        continue
                    '''

                    # 多头开仓
                    if pcon.candicates[period] == 'buy 1':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            _price = self._cmp(ctx.open, ctx.high, pcon.maxmin[period][0])
                            ctx.buy(_price, _unit)
                            self.display(ctx, 'buy', cand)

                            pcon.unit[period] = 1
                            pcon.breakpoint[period].append(pcon.maxmin[period][0])
                            pcon.bpatrhalf[period] = pcon.atrhalf[period]
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1


                    # 多头退市
                    if pcon.candicates[period] == 'sell 2':
                        _unit = sum(pcon._unit[period])
                        _price = self._cmp(ctx.open, ctx.low, pcon.maxmin[period][1])
                        ctx.sell(_price, _unit)
                        self.display(ctx, 'sell', cand)

                        pcon.unit[period] = 0
                        pcon.breakpoint[period] = []
                        pcon.bpatrhalf[period] = 0
                        pcon._unit[period] = []

                        self.freqcount[cand] += 1

                    # 空头开仓
                    if pcon.candicates[period] == 'short 1':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            _price = self._cmp(ctx.open, ctx.low, pcon.maxmin[period][1])
                            ctx.short(_price, _unit)
                            self.display(ctx, 'short', cand)

                            pcon.unit[period] = 1
                            pcon.breakpoint[period].append(pcon.maxmin[period][1])
                            pcon.bpatrhalf[period] = pcon.atrhalf[period]
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1

                    # 空头退市
                    if pcon.candicates[period] == 'cover 2':
                        _unit = sum(pcon._unit[period])
                        _price = self._cmp(ctx.open, ctx.high, pcon.maxmin[period][0])
                        ctx.cover(_price, _unit)
                        self.display(ctx, 'cover', cand)

                        pcon.unit[period] = 0
                        pcon.breakpoint[period] = []
                        pcon.bpatrhalf[period] = 0
                        pcon._unit[period] = []

                        self.freqcount[cand] += 1


                    # 多头加仓
                    if pcon.candicates[period] == 'buy 2':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            p = pcon.breakpoint[period][-1] + pcon.bpatrhalf[period]
                            _price = self._cmp(ctx.open, ctx.high, p)
                            ctx.buy(_price, _unit)
                            self.display(ctx, 'buy', cand)

                            pcon.unit[period] += 1
                            pcon.breakpoint[period].append(pcon.breakpoint[period][-1] + pcon.bpatrhalf[period])
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1


                    # 空头加仓
                    if pcon.candicates[period] == 'short 2':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            p = pcon.breakpoint[period][-1] - pcon.bpatrhalf[period]
                            _price = self._cmp(ctx.open, ctx.low, p)
                            ctx.short(_price, _unit)
                            self.display(ctx, 'short', cand)

                            pcon.unit[period] -= 1
                            pcon.breakpoint[period].append(pcon.breakpoint[period][-1] - pcon.bpatrhalf[period])
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1



                    # 多头止损
                    if pcon.candicates[period] == 'sell 1':
                        _unit = pcon._unit[period][-1]
                        p = pcon.breakpoint[period][-1] - pcon.bpatrhalf[period]
                        _price = self._cmp(ctx.open, ctx.low, p)
                        ctx.sell(_price, _unit)
                        self.display(ctx, 'sell', cand)

                        pcon.unit[period] -= 1
                        pcon.breakpoint[period] = pcon.breakpoint[period][:-1]
                        pcon._unit[period] = pcon._unit[period][:-1]

                        self.freqcount[cand] += 1

                    # 空头止损
                    if pcon.candicates[period] == 'cover 1':
                        _unit = pcon._unit[period][-1]
                        p = pcon.breakpoint[period][-1] + pcon.bpatrhalf[period]
                        _price = self._cmp(ctx.open, ctx.high, p)
                        ctx.cover(_price, _unit)
                        self.display(ctx, 'cover', cand)

                        pcon.unit[period] += 1
                        pcon.breakpoint[period] = pcon.breakpoint[period][:-1]
                        pcon._unit[period] = pcon._unit[period][:-1]

                        self.freqcount[cand] += 1

                pcon.candicates = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}

            self.candicates.clear()




    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless', cand='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  20 Days  **************')
        print ('N20 = %f, N = %f' % (self.n20_mul[cand], self.N_mul[cand][-1]))

        print('%s %s; Number of Units = %d' % (name, cand, self.num_unit_mul[cand]))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        # print (ctx.position(tag, 'BB.SHFE'))
        print ('datetime = %s' % str(ctx.datetime))

        MESSAGE.get_info(cash, name, cand=cand)

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]

# done
class ATRTurtle(Strategy):

    def on_init(self, ctx):
        """初始化数据"""
        '''
        d = {
            '20min': 20, '1h': 60, '1day': 365, '5days': 1825, '20days': 7300, '60days': 21900,
            '10min': 10, '30min': 30, '3h': 180, '2days': 730, '10days': 3650, '30days': 10950,
            '80min': 80, '4h': 240, '4day': 1460,            '80days': 292000, '240days': 87600
        }
        ctx.tt = {}
        for i in d:
            print i
            ctx.tt[i] = Turtle_base(ctx.close, d[i], i, ('r', 'g'))
        '''
        ctx.tt20 = Turtle_base(ctx.close, 20, 'tt20', ('r', 'g'))
        ctx.tt10 = Turtle_base(ctx.close, 10, 'tt10', ('y', 'b'))

        self.N_mul = dict()
        self.N20_mul = dict()
        self.n20_mul = dict()
        self.num_unit_mul = dict()
        self.breakpoint_mul = dict()

        self.risk_ctl = Risk_ctl(ctx)
        self.restraint = Restraint()

        self.L = []

        self.candicates = dict()

        print 'ctx.symbol = %s' % ctx.symbol.split('.')[0]      # eg. 'A1601'

        '''********************************'''

        self.count = Timecount(ctx)
        print self.count

        self.status = {}
        self.timecount = {}
        self.activepcon = Activepcon()
        self.freqcount = {}

    def on_symbol(self, ctx):

        #self.count = Timecount(ctx)

        if ctx.curbar == 1:
            self.N_mul[ctx.symbol] = []
            self.N20_mul[ctx.symbol] = 0.0
            self.n20_mul[ctx.symbol] = 0.0
            self.num_unit_mul[ctx.symbol] = 0
            self.breakpoint_mul[ctx.symbol] = []

            '''ATR type: 1. 20min 2. 1h 3. 1day 4. 5days 5. 20days 6. 60days'''
            self.status[ctx.symbol] = Constatus()
            self.timecount[ctx.symbol] = Timecount(ctx)
            #self.activepcon = Activepcon()
            self.freqcount[ctx.symbol] = 0


        if ctx.curbar > 1:
            self.L.append(max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low)))
            self.N_mul[ctx.symbol] = self.L

            import time

            print ctx.datetime, ctx.open, ctx.symbol, ctx.volume

            isactive = self.timecount[ctx.symbol]._count(ctx)      #return if period active
            self.status[ctx.symbol]._ATR(ctx)

            lst = []
            if isactive:
                self.status[ctx.symbol]._ATR(period=isactive)

                lst = self.activepcon.refresh(ctx.symbol, self.status)

                if '1day' in isactive[0]:
                    for i in self.freqcount:
                        self.freqcount[i] = 0




            '''Strategy start!'''
            for period in ['20min', '1h', '1day', '5days', '20days', '60days']:
                if self.status[ctx.symbol].atrfour[period] and [ctx.symbol, period] in lst:
                    '''write strategy here'''
                    pcon = self.status[ctx.symbol]
                    self.restraint.countall(self.status)

                    #print '^^^^^^^^^^^^^^^^^^^'
                    #print ctx.close[1], pcon.maxmin[period][0], ctx.high
                    stopall = Stopall(self.freqcount[ctx.symbol])

                    stopbuy = Stopbuy(ctxdate=ctx[ctx.symbol].datetime, dateend=ctx.symbol) and\
                              self.restraint.ismaxnuit(ctx.symbol) and stopall


                    if pcon.unit[period] == 0:
                        # 多头入市
                        if ctx.close[1] < (pcon.avgfour[period] + 2 * pcon.atr[period]) and \
                                        ctx.high >= (pcon.avgfour[period] + 2 * pcon.atr[period]) and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'buy 1'
                            pcon.candicates[period] = 'buy 1'
                            self.candicates[ctx.symbol] = True
                            assert pcon.candicates[period] == self.status[ctx.symbol].candicates[period]


                        # 空头入市
                        elif ctx.close[1] > (pcon.avgfour[period] + 2 * pcon.atr[period]) and\
                                        ctx.low <= (pcon.avgfour[period] + 2 * pcon.atr[period]) and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'short 1'
                            pcon.candicates[period] = 'short 1'
                            self.candicates[ctx.symbol] = True
                            assert pcon.candicates[period] == self.status[ctx.symbol].candicates[period]

                    elif 0 < pcon.unit[period] < 4:
                        # 多头加仓
                        if ctx.high >= pcon.breakpoint[period][-1] + pcon.bpatrhalf[period] and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'buy 2'
                            pcon.candicates[period] = 'buy 2'
                            self.candicates[ctx.symbol] = True


                    elif -4 < pcon.unit[period] < 0:
                        # 空头加仓
                        if ctx.low <= pcon.breakpoint[period][-1] - pcon.bpatrhalf[period] and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'short 2'
                            pcon.candicates[period] = 'short 2'
                            self.candicates[ctx.symbol] = True


                    elif 0 < pcon.unit[period]:
                        # 多头退市
                        if (ctx.close[1] > (pcon.avgfour[period] - pcon.atr[period] / 2) >= ctx.low or Timefour(pcon.atrfour_count[period][0])) and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'sell 2'
                            pcon.candicates[period] = 'sell 2'
                            self.candicates[ctx.symbol] = True


                        # 多头止损
                        elif ctx.low <= pcon.breakpoint[period][-1] - pcon.bpatrhalf[period] and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'sell 1'
                            pcon.candicates[period] = 'sell 1'
                            self.candicates[ctx.symbol] = True


                    elif pcon.unit[period] < 0:
                        # 空头退市
                        if (ctx.close[1] < (pcon.avgfour[period] - pcon.atr[period] / 2) <= ctx.high or Timefour(pcon.atrfour_count[period][0])) and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'cover 2'
                            pcon.candicates[period] = 'cover 2'
                            self.candicates[ctx.symbol] = True


                        # 空头止损
                        elif ctx.high >= pcon.breakpoint[period][-1] + pcon.bpatrhalf[period] and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'cover 1'
                            pcon.candicates[period] = 'cover 1'
                            self.candicates[ctx.symbol] = True


    def on_bar(self, ctx):
        if self.candicates:
            print 'self.candicates = %s' % self.candicates

            for cand in self.candicates:
                pcon = self.status[cand]
                for period in ['20min', '1h', '1day', '5days', '20days', '60days']:
                    print('cand = %s' % cand)
                    # cash = ctx.cash()
                    print pcon.candicates

                    # force cover
                    '''
                    if cash < 0:
                        print ('### Current print = %f' % cash)
                        if self.num_unit_mul[cand] > 0:
                            all_units = 0
                            for lst in self.breakpoint_mul[cand]:
                                all_units += lst[1]
                                assert lst[2] == 'buy'
                            self.breakpoint_mul[cand] = []
                            assert isinstance(all_units, int)
                            assert all_units > 0
                            print ('sell_unit = %d' % all_units)
                            _price = self._cmp(ctx.high, ctx.low, ctx.tt10['min'][1])
                            ctx.sell(_price, all_units)
                            self.num_unit_mul[cand]  = 0
                            self.display(ctx, 'sell-force', cand)
                        elif self.num_unit_mul[cand] < 0:
                            all_units = 0
                            for lst in self.breakpoint_mul[cand]:
                                all_units += lst[1]
                                assert lst[2] == 'short'
                            self.breakpoint_mul[cand] = []
                            assert isinstance(all_units, int)
                            assert all_units > 0
                            print ('cover_unit = %d' % all_units)
                            _price = self._cmp(ctx.high, ctx.low, ctx.tt10['max'][1])
                            ctx.cover(_price, all_units)
                            self.num_unit_mul[cand]  = 0
                            self.display(ctx, 'cover-force', cand)

                        continue
                    '''

                    # 多头开仓
                    if pcon.candicates[period] == 'buy 1':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            _price = self._cmp(ctx.open, ctx.high, pcon.maxmin[period][0])
                            ctx.buy(_price, _unit)
                            self.display(ctx, 'buy', cand)

                            pcon.unit[period] = 1
                            pcon.breakpoint[period].append(pcon.maxmin[period][0])
                            pcon.bpatrhalf[period] = pcon.atrhalf[period]
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1


                    # 多头退市
                    if pcon.candicates[period] == 'sell 2':
                        _unit = sum(pcon._unit[period])
                        _price = self._cmp(ctx.open, ctx.low, pcon.maxmin[period][1])
                        ctx.sell(_price, _unit)
                        self.display(ctx, 'sell', cand)

                        pcon.unit[period] = 0
                        pcon.breakpoint[period] = []
                        pcon.bpatrhalf[period] = 0
                        pcon._unit[period] = []

                        self.freqcount[cand] += 1

                    # 空头开仓
                    if pcon.candicates[period] == 'short 1':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            _price = self._cmp(ctx.open, ctx.low, pcon.maxmin[period][1])
                            ctx.short(_price, _unit)
                            self.display(ctx, 'short', cand)

                            pcon.unit[period] = 1
                            pcon.breakpoint[period].append(pcon.maxmin[period][1])
                            pcon.bpatrhalf[period] = pcon.atrhalf[period]
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1

                    # 空头退市
                    if pcon.candicates[period] == 'cover 2':
                        _unit = sum(pcon._unit[period])
                        _price = self._cmp(ctx.open, ctx.high, pcon.maxmin[period][0])
                        ctx.cover(_price, _unit)
                        self.display(ctx, 'cover', cand)

                        pcon.unit[period] = 0
                        pcon.breakpoint[period] = []
                        pcon.bpatrhalf[period] = 0
                        pcon._unit[period] = []

                        self.freqcount[cand] += 1


                    # 多头加仓
                    if pcon.candicates[period] == 'buy 2':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            p = pcon.breakpoint[period][-1] + pcon.bpatrhalf[period]
                            _price = self._cmp(ctx.open, ctx.high, p)
                            ctx.buy(_price, _unit)
                            self.display(ctx, 'buy', cand)

                            pcon.unit[period] += 1
                            pcon.breakpoint[period].append(pcon.breakpoint[period][-1] + pcon.bpatrhalf[period])
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1


                    # 空头加仓
                    if pcon.candicates[period] == 'short 2':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            p = pcon.breakpoint[period][-1] - pcon.bpatrhalf[period]
                            _price = self._cmp(ctx.open, ctx.low, p)
                            ctx.short(_price, _unit)
                            self.display(ctx, 'short', cand)

                            pcon.unit[period] -= 1
                            pcon.breakpoint[period].append(pcon.breakpoint[period][-1] - pcon.bpatrhalf[period])
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1



                    # 多头止损
                    if pcon.candicates[period] == 'sell 1':
                        _unit = pcon._unit[period][-1]
                        p = pcon.breakpoint[period][-1] - pcon.bpatrhalf[period]
                        _price = self._cmp(ctx.open, ctx.low, p)
                        ctx.sell(_price, _unit)
                        self.display(ctx, 'sell', cand)

                        pcon.unit[period] -= 1
                        pcon.breakpoint[period] = pcon.breakpoint[period][:-1]
                        pcon._unit[period] = pcon._unit[period][:-1]

                        self.freqcount[cand] += 1

                    # 空头止损
                    if pcon.candicates[period] == 'cover 1':
                        _unit = pcon._unit[period][-1]
                        p = pcon.breakpoint[period][-1] + pcon.bpatrhalf[period]
                        _price = self._cmp(ctx.open, ctx.high, p)
                        ctx.cover(_price, _unit)
                        self.display(ctx, 'cover', cand)

                        pcon.unit[period] += 1
                        pcon.breakpoint[period] = pcon.breakpoint[period][:-1]
                        pcon._unit[period] = pcon._unit[period][:-1]

                        self.freqcount[cand] += 1

                pcon.candicates = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}

            self.candicates.clear()


    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless', cand='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  20 Days  **************')
        print ('N20 = %f, N = %f' % (self.n20_mul[cand], self.N_mul[cand][-1]))

        print('%s %s; Number of Units = %d' % (name, cand, self.num_unit_mul[cand]))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        # print (ctx.position(tag, 'BB.SHFE'))
        print ('datetime = %s' % str(ctx.datetime))

        MESSAGE.get_info(cash, name, cand=cand)

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]

# TODO predict.price
class ATRBeforehandtt(Strategy):

    def on_init(self, ctx):
        """初始化数据"""
        '''
        d = {
            '20min': 20, '1h': 60, '1day': 365, '5days': 1825, '20days': 7300, '60days': 21900,
            '10min': 10, '30min': 30, '3h': 180, '2days': 730, '10days': 3650, '30days': 10950,
            '80min': 80, '4h': 240, '4day': 1460,            '80days': 292000, '240days': 87600
        }
        ctx.tt = {}
        for i in d:
            print i
            ctx.tt[i] = Turtle_base(ctx.close, d[i], i, ('r', 'g'))
        '''
        ctx.tt20 = Turtle_base(ctx.close, 20, 'tt20', ('r', 'g'))
        ctx.tt10 = Turtle_base(ctx.close, 10, 'tt10', ('y', 'b'))

        self.N_mul = dict()
        self.N20_mul = dict()
        self.n20_mul = dict()
        self.num_unit_mul = dict()
        self.breakpoint_mul = dict()

        self.risk_ctl = Risk_ctl(ctx)
        self.restraint = Restraint()

        self.L = []

        self.candicates = dict()

        print 'ctx.symbol = %s' % ctx.symbol.split('.')[0]      # eg. 'A1601'

        '''********************************'''

        self.count = Timecount(ctx)
        print self.count

        self.status = {}
        self.timecount = {}
        self.activepcon = Activepcon()
        self.freqcount = {}

        self.predict = Predict_module(CTS)

    def on_symbol(self, ctx):

        #self.count = Timecount(ctx)

        if ctx.curbar == 1:
            self.N_mul[ctx.symbol] = []
            self.N20_mul[ctx.symbol] = 0.0
            self.n20_mul[ctx.symbol] = 0.0
            self.num_unit_mul[ctx.symbol] = 0
            self.breakpoint_mul[ctx.symbol] = []

            '''ATR type: 1. 20min 2. 1h 3. 1day 4. 5days 5. 20days 6. 60days'''
            self.status[ctx.symbol] = Constatus()
            self.timecount[ctx.symbol] = Timecount(ctx)
            #self.activepcon = Activepcon()
            self.freqcount[ctx.symbol] = 0


        if ctx.curbar > 1:
            self.L.append(max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low)))
            self.N_mul[ctx.symbol] = self.L

            import time

            print ctx.datetime, ctx.open, ctx.symbol, ctx.volume

            isactive = self.timecount[ctx.symbol]._count(ctx)      #return if period active
            self.status[ctx.symbol]._ATR(ctx)

            lst = []
            if isactive:
                self.status[ctx.symbol]._ATR(period=isactive)

                lst = self.activepcon.refresh(ctx.symbol, self.status)

                if '1day' in isactive[0]:
                    for i in self.freqcount:
                        self.freqcount[i] = 0




            '''Strategy start!'''
            for period in ['20min', '1h', '1day', '5days', '20days', '60days']:
                if self.status[ctx.symbol].atrfour[period] and [ctx.symbol, period] in lst:
                    '''write strategy here'''
                    pcon = self.status[ctx.symbol]
                    self.restraint.countall(self.status)

                    #print '^^^^^^^^^^^^^^^^^^^'
                    #print ctx.close[1], pcon.maxmin[period][0], ctx.high
                    stopall = Stopall(self.freqcount[ctx.symbol])

                    stopbuy = Stopbuy(ctxdate=ctx[ctx.symbol].datetime, dateend=ctx.symbol) and\
                              self.restraint.ismaxnuit(ctx.symbol) and stopall


                    if pcon.unit[period] == 0:
                        # 多头入市
                        if ctx.close[1] < (pcon.avgfour[period] + 2 * pcon.atr[period]) and \
                                self.predict.price(ctx.symbol, ctx.curbar) >= (pcon.avgfour[period] + 2 * pcon.atr[period]) and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'buy 1'
                            pcon.candicates[period] = 'buy 1'
                            self.candicates[ctx.symbol] = True
                            assert pcon.candicates[period] == self.status[ctx.symbol].candicates[period]


                        # 空头入市
                        elif ctx.close[1] > (pcon.avgfour[period] + 2 * pcon.atr[period]) and \
                                self.predict.price(ctx.symbol, ctx.curbar) <= (pcon.avgfour[period] + 2 * pcon.atr[period]) and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'short 1'
                            pcon.candicates[period] = 'short 1'
                            self.candicates[ctx.symbol] = True
                            assert pcon.candicates[period] == self.status[ctx.symbol].candicates[period]

                    elif 0 < pcon.unit[period] < 4:
                        # 多头加仓
                        if self.predict.price(ctx.symbol, ctx.curbar) >= pcon.breakpoint[period][-1] + pcon.bpatrhalf[period] and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'buy 2'
                            pcon.candicates[period] = 'buy 2'
                            self.candicates[ctx.symbol] = True


                    elif -4 < pcon.unit[period] < 0:
                        # 空头加仓
                        if self.predict.price(ctx.symbol, ctx.curbar) <= pcon.breakpoint[period][-1] - pcon.bpatrhalf[period] and \
                                not stopbuy:
                            print '-----------------------'
                            print ctx.symbol, period, 'short 2'
                            pcon.candicates[period] = 'short 2'
                            self.candicates[ctx.symbol] = True


                    elif 0 < pcon.unit[period]:
                        # 多头退市
                        if (ctx.close[1] > (pcon.avgfour[period] - pcon.atr[period] / 2) >= ctx.low or Timefour()) and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'sell 2'
                            pcon.candicates[period] = 'sell 2'
                            self.candicates[ctx.symbol] = True


                        # 多头止损
                        elif ctx.low <= pcon.breakpoint[period][-1] - pcon.bpatrhalf[period] and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'sell 1'
                            pcon.candicates[period] = 'sell 1'
                            self.candicates[ctx.symbol] = True


                    elif pcon.unit[period] < 0:
                        # 空头退市
                        if (ctx.close[1] < (pcon.avgfour[period] - pcon.atr[period] / 2) <= ctx.high or Timefour()) and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'cover 2'
                            pcon.candicates[period] = 'cover 2'
                            self.candicates[ctx.symbol] = True


                        # 空头止损
                        elif ctx.high >= pcon.breakpoint[period][-1] + pcon.bpatrhalf[period] and \
                                not stopall:
                            print '-----------------------'
                            print ctx.symbol, period, 'cover 1'
                            pcon.candicates[period] = 'cover 1'
                            self.candicates[ctx.symbol] = True


    def on_bar(self, ctx):
        if self.candicates:
            print 'self.candicates = %s' % self.candicates

            for cand in self.candicates:
                pcon = self.status[cand]
                for period in ['20min', '1h', '1day', '5days', '20days', '60days']:
                    print('cand = %s' % cand)
                    # cash = ctx.cash()
                    print pcon.candicates

                    # force cover
                    '''
                    if cash < 0:
                        print ('### Current print = %f' % cash)
                        if self.num_unit_mul[cand] > 0:
                            all_units = 0
                            for lst in self.breakpoint_mul[cand]:
                                all_units += lst[1]
                                assert lst[2] == 'buy'
                            self.breakpoint_mul[cand] = []
                            assert isinstance(all_units, int)
                            assert all_units > 0
                            print ('sell_unit = %d' % all_units)
                            _price = self._cmp(ctx.high, ctx.low, ctx.tt10['min'][1])
                            ctx.sell(_price, all_units)
                            self.num_unit_mul[cand]  = 0
                            self.display(ctx, 'sell-force', cand)
                        elif self.num_unit_mul[cand] < 0:
                            all_units = 0
                            for lst in self.breakpoint_mul[cand]:
                                all_units += lst[1]
                                assert lst[2] == 'short'
                            self.breakpoint_mul[cand] = []
                            assert isinstance(all_units, int)
                            assert all_units > 0
                            print ('cover_unit = %d' % all_units)
                            _price = self._cmp(ctx.high, ctx.low, ctx.tt10['max'][1])
                            ctx.cover(_price, all_units)
                            self.num_unit_mul[cand]  = 0
                            self.display(ctx, 'cover-force', cand)

                        continue
                    '''

                    # 多头开仓
                    if pcon.candicates[period] == 'buy 1':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            _price = self._cmp(ctx.open, ctx.high, pcon.maxmin[period][0])
                            ctx.buy(_price, _unit)
                            self.display(ctx, 'buy', cand)

                            pcon.unit[period] = 1
                            pcon.breakpoint[period].append(pcon.maxmin[period][0])
                            pcon.bpatrhalf[period] = pcon.atrhalf[period]
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1


                    # 多头退市
                    if pcon.candicates[period] == 'sell 2':
                        _unit = sum(pcon._unit[period])
                        _price = self._cmp(ctx.open, ctx.low, pcon.maxmin[period][1])
                        ctx.sell(_price, _unit)
                        self.display(ctx, 'sell', cand)

                        pcon.unit[period] = 0
                        pcon.breakpoint[period] = []
                        pcon.bpatrhalf[period] = 0
                        pcon._unit[period] = []

                        self.freqcount[cand] += 1

                    # 空头开仓
                    if pcon.candicates[period] == 'short 1':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            _price = self._cmp(ctx.open, ctx.low, pcon.maxmin[period][1])
                            ctx.short(_price, _unit)
                            self.display(ctx, 'short', cand)

                            pcon.unit[period] = 1
                            pcon.breakpoint[period].append(pcon.maxmin[period][1])
                            pcon.bpatrhalf[period] = pcon.atrhalf[period]
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1

                    # 空头退市
                    if pcon.candicates[period] == 'cover 2':
                        _unit = sum(pcon._unit[period])
                        _price = self._cmp(ctx.open, ctx.high, pcon.maxmin[period][0])
                        ctx.cover(_price, _unit)
                        self.display(ctx, 'cover', cand)

                        pcon.unit[period] = 0
                        pcon.breakpoint[period] = []
                        pcon.bpatrhalf[period] = 0
                        pcon._unit[period] = []

                        self.freqcount[cand] += 1


                    # 多头加仓
                    if pcon.candicates[period] == 'buy 2':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            p = pcon.breakpoint[period][-1] + pcon.bpatrhalf[period]
                            _price = self._cmp(ctx.open, ctx.high, p)
                            ctx.buy(_price, _unit)
                            self.display(ctx, 'buy', cand)

                            pcon.unit[period] += 1
                            pcon.breakpoint[period].append(pcon.breakpoint[period][-1] + pcon.bpatrhalf[period])
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1


                    # 空头加仓
                    if pcon.candicates[period] == 'short 2':
                        _unit = self.risk_ctl.unitctl(ctx, pcon.atrhalf[period])
                        if _unit > 0:
                            p = pcon.breakpoint[period][-1] - pcon.bpatrhalf[period]
                            _price = self._cmp(ctx.open, ctx.low, p)
                            ctx.short(_price, _unit)
                            self.display(ctx, 'short', cand)

                            pcon.unit[period] -= 1
                            pcon.breakpoint[period].append(pcon.breakpoint[period][-1] - pcon.bpatrhalf[period])
                            pcon._unit[period] += [_unit]

                            self.freqcount[cand] += 1



                    # 多头止损
                    if pcon.candicates[period] == 'sell 1':
                        _unit = pcon._unit[period][-1]
                        p = pcon.breakpoint[period][-1] - pcon.bpatrhalf[period]
                        _price = self._cmp(ctx.open, ctx.low, p)
                        ctx.sell(_price, _unit)
                        self.display(ctx, 'sell', cand)

                        pcon.unit[period] -= 1
                        pcon.breakpoint[period] = pcon.breakpoint[period][:-1]
                        pcon._unit[period] = pcon._unit[period][:-1]

                        self.freqcount[cand] += 1

                    # 空头止损
                    if pcon.candicates[period] == 'cover 1':
                        _unit = pcon._unit[period][-1]
                        p = pcon.breakpoint[period][-1] + pcon.bpatrhalf[period]
                        _price = self._cmp(ctx.open, ctx.high, p)
                        ctx.cover(_price, _unit)
                        self.display(ctx, 'cover', cand)

                        pcon.unit[period] += 1
                        pcon.breakpoint[period] = pcon.breakpoint[period][:-1]
                        pcon._unit[period] = pcon._unit[period][:-1]

                        self.freqcount[cand] += 1

                pcon.candicates = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}

            self.candicates.clear()


    def on_exit(self, ctx):
        return

    def display(self, ctx, name='nameless', cand='nameless'):
        cash = ctx.cash()
        #print ('\n')
        print ('**************  20 Days  **************')
        print ('N20 = %f, N = %f' % (self.n20_mul[cand], self.N_mul[cand][-1]))

        print('%s %s; Number of Units = %d' % (name, cand, self.num_unit_mul[cand]))
        print ('current price: %f, cash = %f, equity = %f' % (ctx.close, cash, ctx.equity()))
        if name == 'buy' or name == 'sell' or name == 'sell-force':
            tag = 'long'
        elif name == 'short' or name == 'cover' or name == 'cover-force':
            tag = 'short'
        else:
            tag = 'unkonwn'
        # print (ctx.position(tag, 'BB.SHFE'))
        print ('datetime = %s' % str(ctx.datetime))

        MESSAGE.get_info(cash, name, cand=cand)

    def _cmp(self, a, b, c):
        lst = [a, b, c]
        lst.sort()
        return lst[1]



class RUN(object):
    def __init__(self, cts=CTS):
        from mul_config import ML
        from mul_config import OPEN
        self.rst = []
        self.pf = {}

        '''
        if ML:
            self.rst = [
                ['profile', '20 Days'],
                ['profile4', 'Custom']
            ]
            self.pf = dict()
            set_symbols(cts)
            self.pf['profile'] = add_strategy([DemoStrategy('20 Days')], {'capital': CAPITAL})
            self.pf['profile4'] = add_strategy([MainStrategy('Custom')], {'capital': CAPITAL})
        else:
            self.rst = [
                ['profile', '20 Days'],
                ['profile4', 'Custom']
            ]
            self.pf = dict()
            set_symbols(cts)
            self.pf['profile'] = add_strategy([DemoStrategy('20 Days')], {'capital': CAPITAL})
            self.pf['profile4'] = add_strategy([MainStrategy('Custom')], {'capital': CAPITAL})
        '''

        set_symbols(cts)
        for i in OPEN:
            if OPEN[i]:
                self.rst.append([i, i])
                if i == 'Turtle':
                    self.pf[i] = add_strategy([Turtle(i)], {'capital': CAPITAL})
                elif i == 'Strength Turtle':
                    self.pf[i] = add_strategy([Strengthtt(i)], {'capital': CAPITAL})
                elif i == 'Beforehand Turtle':
                    self.pf[i] = add_strategy([Beforehandtt(i)], {'capital': CAPITAL})
                elif i == 'ATR Turtle':
                    self.pf[i] = add_strategy([ATRTurtle(i)], {'capital': CAPITAL})
                elif i == 'ATR Beforehand Turtle':
                    self.pf[i] = add_strategy([ATRBeforehandtt(i)], {'capital': CAPITAL})

        assert len(self.rst)




    def results(self):
        start = timeit.default_timer()
        run()
        stop = timeit.default_timer()
        print "运行耗时: %d秒" % (stop - start)

        filename = './Output/' + 'RESULT' + '.txt'
        f = open(filename, 'w')

        MESSAGE.display()

        for pf in self.rst:
            curve = finance.create_equity_curve(self.pf[pf[0]].all_holdings())
            print pf[1], finance.summary_stats(curve, 252 * 4 * 60)

            f.write('\n')
            f.write(pf[1])
            f.write('\n')
            f.write(str(finance.summary_stats(curve, 252 * 4 * 60)))

            # figname = 'K-line.jpg'

            # K-line
            # myplot.plot_strategy(self.pf[pf[0]].data(0), self.pf[pf[0]].technicals(0), self.pf[pf[0]].deals(0),
                                 # curve.equity, figname)

            # Equity-curve

            myplot.plot_curves([curve.equity], colors=['r'], names=[self.pf[pf[0]].name(0) + '-equity'])

            # myplot.plot_curves([curve.cash], colors=['g'], names=[self.pf[pf[0]].name(0) + '-cash'])
            # myplot.plot_curves([curve.returns], colors=['b'], names=[self.pf[pf[0]].name(0) + '-returns'])
            # myplot.plot_curves([curve.equity, curve.cash, curve.returns, curve.commission],
            #                    colors=['r', 'g', 'b', 'y'],
            #                    names=[self.pf[pf[0]].name(0)])

        f.close()



if __name__ == '__main__':

    # Reference.check()
    time.sleep(5)
    P = RUN()
    P.results()
    myplot.show()
