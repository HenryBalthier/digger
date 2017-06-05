# -*- coding: utf-8 -*-

import csv
import math
import os
import mychoice
from MY_digger.pred_modu_inf import Predict
from mul_config import Default
from mul_config import PERIOD

CTS = Default.get_name()
RSK = Default.get_risk()
CLEVEL = Default.get_clevel()

FILTER = 0
OPENPRED = 0

yellow = '\033[1;33m'
yellow_ = '\033[0m'


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
        '''
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
        '''


class Operation_cycle(object):
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
    def __init__(self):
        self.p = Predict()
        self.p.run()
        # print (self.p.dat_output[1])

    def signal(self, index):
        if not OPENPRED:
            return 0
        elif self.price(index) * self.confidence(index) > 0:
            return 1
        elif self.price(index) * self.confidence(index) <= 0:
            return -1
        else:
            assert False

    def price(self, index):
        if self.p.dat_output[index - 1] > 0:
            return 1
        elif self.p.dat_output[index - 1] < 0:
            return -1
        else:
            return 0

    def confidence(self, index):
        if self.p.confidence[index - 1] > CLEVEL:
            return 1
        else:
            return 0


class Risk_ctl(object):
    def __init__(self, ctx):
        risk = Get_pcon()
        print ctx.symbol
        row = risk.csv_test(ctx.symbol.split('.')[0])
        assert len(row) == 1
        print row
        self._ratio = row[0][4]
        self._tick = row[0][6]
        self._vol = row[0][7]

    def unitctl(self, ctx, N20, name='nameless'):
        cash = ctx.cash()
        # cash = Default.get_capital()
        _ratio = float(self._ratio)
        _vol = int(self._vol)
        _unit = math.floor((cash * RSK * _ratio) / (N20 * _vol))
        assert cash > (_unit * _vol * 2 * N20 / _ratio)
        print yellow
        print ('_unit = %d, real_unit = %f' % (_unit, (cash * RSK * _ratio) / (N20 * _vol)))
        print yellow_
        return int(_unit)


class Get_pcon(object):
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
        with open('./data/contracts.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[self.cts['code']] == code:
                    n.append(row)
        return n


__all__ = ['Operation_cycle', 'Predict_module', 'Risk_ctl', 'Reference',
           'CTS', 'CLEVEL', 'RSK', 'FILTER']

if __name__ == '__main__':
    r = Reference()
    r.check()
