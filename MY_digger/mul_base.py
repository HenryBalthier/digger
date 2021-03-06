# -*- coding: utf-8 -*-

import csv
import math
import os
# import mychoice
import MY_digger.mychoice
# from MY_digger.mul_pred_modu_inf import Predict
from mul_config import Default
from mul_config import PERIOD
from mul_pred_modu import Predict

CTS = Default.get_name()        # 合约名
RSK = Default.get_risk()        # 风控
CLEVEL = Default.get_clevel()   # 置信度
AGENCY = Default.get_agency()   # 交易所？
STOPBUY = Default.get_stopbuy() #
A3 = 10     # 反向涨跌阈值

FILTER = 0
OPENPRED = 1                    # 1开启预测模块

yellow = '\033[1;33m'
yellow_ = '\033[0m'

FLAG = {
            '20min': False,
            '1h': False,
            '1day': False,
            '5days': False,
            '20days': False,
            '60days': False
}

FREQCOUNT = 300         # 高频交易最高次数

"""
class Reference(object):
    @classmethod
    def check(cls):
        testdir = Default.get_testdir()
        filename = []
        for i in testdir:
            filename.append(i.split('/')[2] + '_' + i.split('/')[4].split('.')[0] + '.h5')
        lst = [
            Default.get_testdir(),
            Default.get_traindir(),
            Default.get_sadir(),
            filename
        ]

        x = dict()
        print yellow
        for i in lst:
            for j in i:
                x[j] = os.path.exists(j)
                print '%s exists : \n' % j, x[j]
        print yellow_

        for i in x:
            assert x[i] is True

        # fix

        for i in lst:
            for j in i:
                if not x[j]:
                    if i == Default.get_testdir():
                        for k in Default.get_datadir():
                            if os.path.exists(k):
                                import mul_utils
                                print yellow
                                print 'fixing path: %s' % j
                                print yellow_
                                r = mul_utils.SplitFile()
                                r.split()
                            else:
                                assert i

                    if i == Default.get_traindir():
                        if os.path.exists(Default.get_datadir()):
                            import mul_utils
                            print yellow
                            print 'fixing path: %s' % j
                            print yellow_
                            r = mul_utils.SplitFile()
                            r.split()
                        else:
                            assert i

                    if i == Default.get_sadir():
                        import news_infer
                        print yellow
                        print 'fixing path: %s' % j
                        print yellow_
                        news_infer.SAinf.inf()

                    if i == filename:
                        import pred_modu
                        print yellow
                        print 'fixing path: %s' % j
                        print yellow_
                        t = pred_modu.Train()
                        t.run()
"""


class Operation_cycle(object):
    """操作周期，当前版本已取消该功能"""
    def __init__(self):
        self.count = 0
        self.cycle = Default.get_opcycle() - 1
        if self.cycle < 0:
            self.cycle = 0
        if PERIOD[-3:-1] == 'DAY':
            self.cycle = 0

    def counter(self, reset=FILTER):
        if self.count < self.cycle:
            self.count += 1
            return 0
        else:
            if reset:
                self.count = 0
            return 1


class Predict_module(object):
    """结合机器学习预测指标"""
    def __init__(self, cts):
        print cts
        lst = []
        for i in cts:
            lst.append(i.split('.')[0])

        self.p = Predict(lst)
        self.p.run()
        # print (self.p.dat_output[1])

    def signal(self, ctxsymbol, index):
        """力度指标"""
        from mul_config import PRED

        ctxsymbol = ctxsymbol.split('.')[0]
        if not PRED:
            return 0
        elif self.pricesymbel(ctxsymbol, index) * self.confidence(ctxsymbol, index) > 0:
            return 1
        elif self.pricesymbel(ctxsymbol, index) * self.confidence(ctxsymbol, index) < 0:
            return -1
        elif self.pricesymbel(ctxsymbol, index) * self.confidence(ctxsymbol, index) == 0:
            return 0
        else:
            assert False

    def pricesymbel(self, ctxsymbol, index):
        """走势"""
        if self.p.dat_output[ctxsymbol][index - 1] > 0:
            return 1
        elif self.p.dat_output[ctxsymbol][index - 1] < 0:
            return -1
        else:
            return 0

    def confidence(self, ctxsymbol, index):
        """置信度"""
        if self.p.confidence[ctxsymbol][index - 1] > CLEVEL:
            return 1
        else:
            return 0

    #@TODO
    def price(self, symbol, index):
        """具体价格预测"""
        return True



class Risk_ctl(object):
    """风控，头寸管理"""
    def __init__(self, ctx):
        risk = Get_pcon()
        print ctx.symbol
        row = risk.csv_test(ctx.symbol.split('.')[0])
        assert len(row) == 1
        self._ratio = row[0][4]
        self._tick = row[0][6]
        self._vol = row[0][7]

    def unitctl(self, ctx, N20, name='nameless'):
        cash = ctx.cash()
        if not cash > 0:
            print 'cash = %d' % cash
            raise AssertionError
        # cash = Default.get_capital()
        _ratio = float(self._ratio) * AGENCY
        _vol = int(self._vol)
        _unit = math.floor((cash * RSK * _ratio) / (N20 * _vol))
        if not cash > (_unit * _vol * 2 * N20 / _ratio):
            print 'cash = %d' % cash
            print 'unit = %d' % _unit
            print ' (_unit * _vol * 2 * N20 / _ratio) = %d' % (_unit * _vol * 2 * N20 / _ratio)
            raise AssertionError
        print yellow
        print ('_unit = %d, real_unit = %f' % (_unit, (cash * RSK * _ratio) / (N20 * _vol)))
        print yellow_
        return int(_unit)


class Get_pcon(object):
    """合约信息获取"""
    def __init__(self):
        self.cts = {
            'code': 0,
            'exchange': 1,
            'name': 2,
            'spell': 3,
            'long_margin_ratio': 4,
            'short_margin_ratio': 5,
            'price_tick': 6,
            'volume_multiple': 7
            }

    def csv_test(self, code=None):
        n = []
        code = code.split('1')[0]
        with open('./data/contracts.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[self.cts['code']] == code:
                    n.append(row)
        return n


class Restraint(object):
    """交易约束"""
    def __init__(self):
        from mul_config import LIST, RANGE, ID
        self.r = Default.get_restraint()
        self.num_up = {}
        self.num_down = {}
        self.lst = LIST(RANGE, ID)

        print self.r.high
        print self.r.low

        self.sum_up = 0
        self.sum_down = 0

        for pcon in self.lst:
            pcon2 = pcon.split('1')[0]
            for i in self.r.high:
                if pcon2 in self.r.high[i]:
                    print '%s in high correlation [%s]' % (pcon, i)
                    self.num_up[i] = 0
                    self.num_down[i] = 0

            for i in self.r.low:
                if pcon2 in self.r.low[i]:
                    print '%s in low correlation [%s]' % (pcon, i)
                    self.num_up[i] = 0
                    self.num_down[i] = 0

    def countall(self, status):
        self.countclear()
        for pcon in status:
            for j in ['20min', '1h', '1day', '5days', '20days', '60days']:
                if status[pcon].unit[j] > 0:
                    self.sum_up += status[pcon].unit[j]
                elif status[pcon].unit[j] < 0:
                    self.sum_down += status[pcon].unit[j]


            pcon2 = pcon.split('1')[0]
            for i in self.r.high:
                if pcon2 in self.r.high[i]:
                    print '%s in high correlation [%s]' % (pcon, i)
                    for j in ['20min', '1h', '1day', '5days', '20days', '60days']:
                        if status[pcon].unit[j] > 0:
                            self.num_up[i] += status[pcon].unit[j]
                        elif status[pcon].unit[j] < 0:
                            self.num_down[i] += status[pcon].unit[j]
            for i in self.r.low:
                if pcon2 in self.r.low[i]:
                    print '%s in low correlation [%s]' % (pcon, i)
                    for j in ['20min', '1h', '1day', '5days', '20days', '60days']:
                        if status[pcon].unit[j] > 0:
                            self.num_up[i] += status[pcon].unit[j]
                        elif status[pcon].unit[j] < 0:
                            self.num_down[i] += status[pcon].unit[j]


    def countclear(self):
        self.sum_up = 0
        self.sum_down = 0

        for i in self.r.high:
            self.num_up[i] = 0
            self.num_down[i] = 0

        for i in self.r.low:
            self.num_up[i] = 0
            self.num_down[i] = 0

    def ismaxnuit(self, pcon):
        pcon2 = pcon.split('1')[0]
        for i in self.r.high:
            if pcon2 in self.r.high[i]:
                if self.num_up[i] >= 6 or self.num_down[i] <= -6:
                    print '%s >> MAXUNIT' % pcon
                    return False
        for i in self.r.low:
            if pcon2 in self.r.low[i]:
                if self.num_up[i] >= 8 or self.num_down[i] <= -8:
                    print '%s >> MAXUNIT' % pcon
                    return False
        if self.sum_up >= 12 or self.sum_down <= -12:
            print '%s >> MAXUNIT' % pcon
            return False
        return True


    '''
    def count(self, pcon, num_unit_mul, trend):
        pcon2 = pcon.split('1')[0]
        for i in self.r.high:
            if pcon2 in self.r.high[i]:
                print '%s in HIGH correlation [%s]' % (pcon, i)
                if trend > 0:
                    print 'The total units in [%s]_up is %d' % (i, self.num_up[i])
                    if self.num_up[i] < 6 and self.count_all(num_unit_mul, trend):
                        self.num_up[i] += 1
                        print 'Now units num + 1 = %d' % self.num_up[i]
                    else:
                        print '***  The units are FULL !   ***\n'
                        return False
                elif trend < 0:
                    print 'The total units in [%s]_down is %d' % (i, self.num_down[i])
                    if self.num_down[i] > -6 and self.count_all(num_unit_mul, trend):
                        self.num_down[i] -= 1
                        print 'Now units num - 1 = %d' % self.num_down[i]
                    else:
                        print '***   The units are FULL !   ***\n'
                        return False

        for j in self.r.low:
            if pcon2 in self.r.low[j]:
                print '%s in LOW correlation [%s]' % (pcon, j)
                if trend > 0:
                    print 'The total units in [%s]_up is %d' % (j, self.num_up[j])
                    if self.num_up[j] < 10 and self.count_all(num_unit_mul, trend):
                        self.num_up[j] += 1
                        print 'Now units num + 1 = %d' % self.num_up[j]
                    else:
                        print '***   The units are FULL !   ***\n'
                        return False
                elif trend < 0:
                    print 'The total units in [%s]_down is %d' % (j, self.num_down[j])
                    if self.num_down[j] > -10 and self.count_all(num_unit_mul, trend):
                        self.num_down[j] -= 1
                        print 'Now units num - 1 = %d' % self.num_down[j]
                    else:
                        print '***   The units are FULL !   ***\n'
                        return False
        return True

    def count_clear(self, symbol, num_unit_mul):
        pcon = symbol.split('.')[0]
        pcon = pcon.split('1')[0]
        for i in self.r.high:
            if pcon in self.r.high[i]:
                print '%s in HIGH correlation [%s]' % (pcon, i)
                if num_unit_mul[symbol] > 0:
                    self.num_up[i] -= num_unit_mul[symbol]
                    print 'The total units in [%s]_up is %d' % (i, self.num_up[i])
                else:
                    self.num_down[i] -= num_unit_mul[symbol]
                    print 'The total units in [%s]_up is %d' % (i, self.num_up[i])

        for j in self.r.low:
            if pcon in self.r.low[j]:
                print '%s in LOW correlation [%s]' % (pcon, j)
                if num_unit_mul[symbol] > 0:
                    self.num_up[j] -= num_unit_mul[symbol]
                    print 'The total units in [%s]_up is %d' % (j, self.num_up[j])
                else:
                    self.num_down[j] -= num_unit_mul[symbol]
                    print 'The total units in [%s]_up is %d' % (j, self.num_up[j])

    def count_all(self, num_unit_mul, trend):
        sum_up = 0
        sum_down = 0
        for i in num_unit_mul:
            if num_unit_mul[i] > 0:
                sum_up += num_unit_mul[i]
            else:
                sum_down += num_unit_mul[i]

        print 'sum_up = %d; sum_down = %d' % (sum_up, sum_down)
        if trend > 0 and sum_up == 12:
            print '***   The TOTAL units are FULL !   ***\n'
            return False
        elif trend < 0 and sum_down == -12:
            print '***   The TOTAL units are FULL !   ***\n'
            return False
        else:
            return True
    '''

class Messages(object):
    """交易信息打印"""
    def __init__(self, init_cash):
        self.cash = [init_cash]
        self.operation = {
            'buy': 0,
            'sell': 0,
            'sell-force': 0,
            'short': 0,
            'cover': 0,
            'cover-force': 0
        }
        self.cand = {}

    def get_info(self, cash, name, cand='None'):
        self.cash.append(cash)
        self.operation[name] += 1
        if cand not in self.cand:
            self.cand[cand] = {
                'buy': 0,
                'sell': 0,
                'sell-force': 0,
                'short': 0,
                'cover': 0,
                'cover-force': 0
            }
        self.cand[cand][name] += 1

    def display(self):
        print yellow
        print 'Detals:'
        print 'Total operations:'

        print self.cand
        print self.operation
        print yellow_

        # print self.cash
        print '\n'


def Stopbuy(ctxdate, dateend):
    """
    风控
    eg
    2016-01-18 10:44:00
    C1701.SHFE
    """
    print ctxdate
    print dateend

    y = str(ctxdate)[2:4]
    m = str(ctxdate)[5:7]
    ys = dateend.split('.')[0][-4:-2]
    ms = dateend.split('.')[0][-2:]
    # print(int(y), int(m), int(ys), int(ms))

    if y == ys and m == ms:
        print '%s >> STOPBUY' % dateend
        return True

    return False

def Stopall(freqcount):
    """防止高频的风控"""
    if freqcount > FREQCOUNT:
        print '%s >> freqcount' % freqcount
        return True
    return False

class Activepcon(object):
    """to active some pcon by sort volume"""
    # @TODO
    def __init__(self):
        self.pconact = {}

    def refresh(self, symbol, status):
        from mul_config import MAXCONTRACT
        print '%s.vol = %s' % (symbol, status[symbol].vol)
        self.pconact[symbol] = status[symbol].vol
        l = []
        d = {}
        for i in self.pconact:
            for j in ['20min', '1h', '1day', '5days', '20days', '60days']:
                l.append(self.pconact[i][j] + 1000000 * abs(status[i].unit[j]))

                if l[-1] != 0:
                    d[l[-1]] = [i, j]
        l = sorted(l, reverse=True)
        self.level = l[-1] if len(l) <= MAXCONTRACT else l[MAXCONTRACT-1]
        print 'l = %s' % l
        lst = []
        for i in l:
            if i != 0:
                print d.get(i, None)
                lst.append(d.get(i, None))
        return lst

    def sort(self):
        pass

class Timecount(object):
    """to count 6 type of ATR"""
    # @TODO
    def __init__(self, ctx):
        self.count = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}
        self.count_half = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}
        self.count_four = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}

        date = str(ctx.datetime)
        self.preday = date.split(' ')[0].split('-')[2]
        self.prehour = date.split(' ')[1].split(':')[0]
        self.premin = date.split(' ')[1].split(':')[1]
        print date
        print self.preday   # eg '2016-11-30 20:59:00'

    def _count(self, ctx):
        date = str(ctx.datetime)
        today = date.split(' ')[0].split('-')[2]
        curhour = date.split(' ')[1].split(':')[0]
        curmin = date.split(' ')[1].split(':')[1]
        if today != self.preday:
            for i in ['1day', '5days', '20days', '60days']:
                self.count[i] += 1
                self.count_four[i] += 1
                if i != '1day':
                    self.count_half[i] += 1
            self.preday = today
        elif curhour != self.prehour:
            self.count['1h'] += 1
            self.count_four['1h'] += 1
            self.count_half['1day'] += 1
            self.prehour = curhour

        if self.premin != curmin:
            self.count['20min'] += 1
            self.count_four['20min'] += 1
            self.count_half['20min'] += 1
            self.count_half['1h'] += 1

        self.premin = curmin
        return self._ifactive()


    def _ifactive(self):
        """['20min', '1h', '1day', '5days', '20days', '60days']"""

        global FLAG
        '''count ATR1'''
        s = set()

        if self.count['20min'] == 20:
            s.add('20min')
            self.count['20min'] = 0

        if self.count['1h'] == 1:
            s.add('1h')
            self.count['1h'] = 0

        if self.count['1day'] == 1:
            s.add('1day')
            self.count['1day'] = 0

        if self.count['5days'] == 5:
            s.add('5days')
            self.count['5days'] = 0

        if self.count['20days'] == 20:
            s.add('20days')
            self.count['20days'] = 0

        if self.count['60days'] == 60:
            s.add('60days')
            self.count['60days'] = 0

        '''count ATR0.5'''
        s_half = set()
        if self.count_half['20min'] == 10:
            s_half.add('20min')
            self.count_half['20min'] = 0

        if self.count_half['1h'] == 30:
            s_half.add('1h')
            self.count_half['1h'] = 0

        if self.count_half['1day'] == 3:
            s_half.add('1day')
            self.count_half['1day'] = 0

        if self.count_half['5days'] == 2:
            s_half.add('5days')
            self.count_half['5days'] = 0

        if self.count_half['20days'] == 10:
            s_half.add('20days')
            self.count_half['20days'] = 0

        if self.count_half['60days'] == 30:
            s_half.add('60days')
            self.count_half['60days'] = 0

        '''count ATR4'''
        s_four = set()

        if self.count_four['20min'] == 80:
            s_four.add('20min')
            self.count_four['20min'] = 0

        if self.count_four['1h'] == 4:
            s_four.add('1h')
            self.count_four['1h'] = 0

        if self.count_four['1day'] == 4:
            s_four.add('1day')
            self.count_four['1day'] = 0

        if self.count_four['5days'] == 20:
            s_four.add('5days')
            self.count_four['5days'] = 0

        if self.count_four['20days'] == 80:
            s_four.add('20days')
            self.count_four['20days'] = 0

        if self.count_four['60days'] == 240:
            s_four.add('60days')
            self.count_four['60days'] = 0
        #if self.count[]
        if len(s):
            print 's = %s' % s
        if len(s_half):
            print 's_half = %s' % s_half
        if len(s_four):
            print 's_four = %s' % s_four

        if len(s) or len(s_half) or len(s_four):
            return [s, s_half, s_four]
        else:
            return None


    def _run(self, period):
        # @TODO run ATR and common
        pass


class Constatus(object):
    """合约信息存储"""
    def __init__(self):
        self.status = {}
        self.status = {
            'activation': set(),
            'units': [0, 0, 0, 0, 0, 0],
            'breakpoint': [[], [], [], [], [], []],
            'stoploss': [[], [], [], [], [], []],
            'nextbp': [],
            'stopbuy': False
        }

        '''Status'''
        self.unit = {'20min': 0,'1h': 0,'1day': 0,'5days': 0,'20days': 0,'60days': 0}
        self.breakpoint = {'20min': [],'1h': [],'1day': [],'5days': [],'20days': [],'60days': []}
        self.bpatrhalf = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}
        self.candicates = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}

        self._unit = {'20min': [],'1h': [],'1day': [],'5days': [],'20days': [],'60days': []}

        self.atr = {'20min': 0,'1h': 0,'1day': 0,'5days': 0,'20days': 0,'60days': 0}
        self.vol = {'20min': 0,'1h': 0,'1day': 0,'5days': 0,'20days': 0,'60days': 0}

        self.atr_count = {'20min': [0, 1],'1h': [0, 1],'1day': [0, 1],'5days': [0, 1],'20days': [0, 1],'60days': [0, 1]}
        self.vol_count = {'20min': [0, 1],'1h': [0, 1],'1day': [0, 1],'5days': [0, 1],'20days': [0, 1],'60days': [0, 1]}

        self.atrhalf = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}
        self.atrfour = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}
        self.atrhalf_count = {'20min': [0, 1], '1h': [0, 1], '1day': [0, 1], '5days': [0, 1], '20days': [0, 1],
                          '60days': [0, 1]}
        self.atrfour_count = {'20min': [0, 1], '1h': [0, 1], '1day': [0, 1], '5days': [0, 1], '20days': [0, 1],
                          '60days': [0, 1]}

        self.maxmin = {'20min': [0, 0], '1h': [0, 0], '1day': [0, 0], '5days': [0, 0], '20days': [0, 0],
                              '60days': [0, 0]}
        self.maxminhalf = {'20min': [0, 0], '1h': [0, 0], '1day': [0, 0], '5days': [0, 0], '20days': [0, 0],
                       '60days': [0, 0]}
        self.maxminfour = {'20min': [0, 0], '1h': [0, 0], '1day': [0, 0], '5days': [0, 0], '20days': [0, 0],
                       '60days': [0, 0]}

        self.maxmin_count = {'20min': [0, 9999], '1h': [0, 9999], '1day': [0, 9999], '5days': [0, 9999],
                             '20days': [0, 9999], '60days': [0, 9999]}
        self.maxminhalf_count = {'20min': [0, 9999], '1h': [0, 9999], '1day': [0, 9999], '5days': [0, 9999],
                             '20days': [0, 9999], '60days': [0, 9999]}
        self.maxminfour_count = {'20min': [0, 9999], '1h': [0, 9999], '1day': [0, 9999], '5days': [0, 9999],
                             '20days': [0, 9999], '60days': [0, 9999]}

        self.avgfour = {'20min': 0, '1h': 0, '1day': 0, '5days': 0, '20days': 0, '60days': 0}
        self.avgfour_count = {'20min': [0, 1], '1h': [0, 1], '1day': [0, 1], '5days': [0, 1], '20days': [0, 1],
                              '60days': [0, 1]}



    def _record(self, ctx, opt=None):
        """record status in every minutes"""
        if opt:
            for i in opt:
                assert i in ['activation', 'units', 'breakpoint', 'stoploss', 'nextbp', 'stopbuy']


    def _ATR(self, ctx=None, period=None):
        """compute ATR in 6 period"""
        if period:
            '''compute ATR and VOL in different period'''
            if len(period[0]):
                for p in period[0]:
                    self.atr[p] = self.atr_count[p][0] / self.atr_count[p][1]
                    self.atr_count[p][0] = self.atr_count[p][1] = 1
                    self.vol[p] = self.vol_count[p][0] / self.vol_count[p][1]
                    self.vol_count[p][0] = self.vol_count[p][1] = 1

                    self.maxmin[p] = self.maxmin_count[p][:]
                    self.maxmin_count[p] = [0, 9999]
                    assert self.maxmin[p] != self.maxmin_count[p]
                print 'maxmin = %s' % self.maxmin

            if len(period[1]):
                for p in period[1]:
                    self.atrhalf[p] = self.atrhalf_count[p][0] / self.atrhalf_count[p][1]
                    self.atrhalf_count[p][0] = self.atrhalf_count[p][1] = 1

                    self.maxminhalf[p] = self.maxminhalf_count[p][:]
                    self.maxminhalf_count[p] = [0, 9999]
                    assert self.maxminhalf[p] != self.maxminhalf_count[p]

            if len(period[2]):
                for p in period[2]:
                    self.atrfour[p] = self.atrfour_count[p][0] / self.atrfour_count[p][1]
                    self.atrfour_count[p][0] = self.atrfour_count[p][1] = 1

                    self.maxminfour[p] = self.maxminfour_count[p][:]
                    self.maxminfour_count[p] = [0, 9999]
                    assert self.maxminfour[p] != self.maxminfour_count[p]

                    self.avgfour[p] = self.avgfour_count[p][0] / self.avgfour_count[p][1]
                    self.avgfour_count[p][0] = self.avgfour_count[p][1] = 1

            # print 'atr = %s' % self.atr
        else:
            '''count ATR and VOL in every minute'''
            N = max((ctx.high - ctx.close[1]), (ctx.close[1] - ctx.low), (ctx.high - ctx.low))
            MAX = ctx.high[1]
            MIN = ctx.low[1]
            vol = ctx.volume
            avg = (ctx.high[1] + ctx.close[1]) / 2
            for i in self.atr_count:
                self.atr_count[i][0] += N
                self.atr_count[i][1] += 1

                self.avgfour_count[i][0] += avg
                self.avgfour_count[i][1] += 1

                self.atrhalf_count[i][0] += N
                self.atrhalf_count[i][1] += 1
                self.atrfour_count[i][0] += N
                self.atrfour_count[i][1] += 1

                self.vol_count[i][0] += vol
                self.vol_count[i][1] += 1

                if MAX > self.maxmin_count[i][0]:
                    self.maxmin_count[i][0] = MAX
                if MIN < self.maxmin_count[i][1]:
                    self.maxmin_count[i][1] = MIN

                if MAX > self.maxminhalf_count[i][0]:
                    self.maxminhalf_count[i][0] = MAX
                if MIN < self.maxminhalf_count[i][1]:
                    self.maxminhalf_count[i][1] = MIN

                if MAX > self.maxminfour_count[i][0]:
                    self.maxminfour_count[i][0] = MAX
                if MIN < self.maxminfour_count[i][1]:
                    self.maxminfour_count[i][1] = MIN


def BackStrengh(atrhalf):

    if atrhalf > A3:
        return True
    else:
        return False

def Timefour(atrfour_count):
    if atrfour_count == 1:
        return True
    else:
        return False


__all__ = [
    'Operation_cycle',
    'Predict_module',
    'Risk_ctl',
    # 'Reference',
    'CTS',
    'CLEVEL',
    'RSK',
    'FILTER',
    'Restraint',
    'Messages',
    'Stopbuy',
    'Stopall',
    'Activepcon',
    'Timecount',
    'Constatus',
    'Timefour',
    'BackStrengh'
]

if __name__ == '__main__':
    lst = ['A', 'B', 'C', 'CU', 'AG']
    dct = {'A': 0, 'B': 0, 'CU': 0, 'AG': 0}
    r = Restraint(lst)
    for i in range(15):
        if dct['A'] < 4:
            if r.count('A', dct, 1):
                dct['A'] += 1

    for i in range(15):
        if dct['B'] < 4:
            if r.count('B', dct, 1):
                dct['B'] += 1

    for i in range(15):
        if dct['CU'] < 4:
            if r.count('CU', dct, 1):
                dct['CU'] += 1

    for i in range(15):
        if dct['AG'] > -4:
            if r.count('AG', dct, -1):
                dct['AG'] -= 1

    r.count_clear('A', dct)
    dct['A'] = 0

    for i in range(15):
        if dct['B'] < 4:
            if r.count('B', dct, 1):
                dct['B'] += 1
