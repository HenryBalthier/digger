# -*- coding: utf-8 -*-

ML = 0                   # N Predict
PRED = 0                 # only price predict

OPEN = {                                # 开启各个策略
    'Turtle':                   0,
    'Strength Turtle':          1,
    'Beforehand Turtle':        0,
    'ATR Turtle':               0,
    'ATR Beforehand Turtle':    0
}

# 关键参数
PCON = 'AUTO'            # 合约简称
# PCON = 'CUSTOM'
# PCON = ['A', 'B']
EXCHANGE = 'SHFE'
PERIOD = '1MINUTE'         # 合约数据时间格式
RANGE = ['C']               # 合约编号
ID = ['1701', '1703']       # 合约日期
MAXCONTRACT = 3             # 最多同时交易合约数

AGENCY = 1
RANDOM = 1                  # 1开启随即挂单失败

# DATESTART = '2015-01-01 00:00:00'
# DATEEND = '2015-12-01 00:00:00'
# STOPBUY = '2015-12-10 00:00:00'
# STOPBUY = 7

# 回测相关参数
CAPITAL = 1000000       # 初始资金总数
RISK = 0.05             # 承担风险指数

CLEVEL = 60              # 机器学习预测置信比

OPCYCLE = 4             # 操作周期 = OPCYCLE * 数据最小时间单位

# 自定义策略相关参数
MAXUNIT = 4             # 最大头寸数限制
TT = {                  # 海龟相关数据（单位时间倍率）
    'IN': 20,           # 入市指标20
    'OUT': 10,          # 退市指标10
    'N': 20             # 波动率20
}

'''Remove ML para from config'''

# 机器学习相关参数
CLOSE_IDX = 1
PRED_RANGE = 5
SPLIT_RATIO = 0.2
NUM_STEPS = 1

# 态势分析相关参数
CLOSE_IDX = 1
HIGH_IDX = 2
LOW_IDX = 3
PRED_RANGE = 5
SPLIT_RATIO = 0.2
PCON_DICT = {'A': 'a', 'AG': '豆', 'B': '螺纹钢', 'BB': 'bb', 'C': 'c'}


# Params of PCON
HIGHRIS = {
            'Bean': ['B', 'BB', 'A'],
            'Oil': [],
            'Gold & Silver': []
        }
LOWRIS = {
            'Copper like': ['CU', 'AG'],
            'Grain': []
        }
OTHERS = {
            'other': []
        }


def LIST(range, id):
    """First step to choice contracts"""
    # @TODO add some restriction
    lst = []
    for r in range:
        for i in id:
            lst.append(r + i)

    return lst


class Default(object):

    @classmethod
    def get_datadir(cls):
        f = cls.get_futures()
        data_dir = []
        for s in f.pcon:
            data_dir.append('./traindata/' + PERIOD + '/' + EXCHANGE + '/' + s + '_full.csv')
        return data_dir

    '''
    @classmethod
    def get_traindir(cls):
        pcon = cls.get_choice()
        train_dir = []
        for s in pcon:
            train_dir.append('./traindata/' + PERIOD + '/' + EXCHANGE + '/' + s + '.train.csv')
        return train_dir

    @classmethod
    def get_testdir(cls):
        pcon = cls.get_choice()
        test_dir = []
        for s in pcon:
            test_dir.append('./data/' + PERIOD + '/' + EXCHANGE + '/' + s + '.csv')
        return test_dir

    @classmethod
    def get_sadir(cls):
        pcon = cls.get_choice()
        sa_dir = []
        for s in pcon:
            sa_dir.append('./traindata/' + PERIOD + '/' + EXCHANGE + '/' + s + '.sa.csv')
        return sa_dir
    '''

    @classmethod
    def get_name(cls):
        futures = cls.get_futures()

        print 'name=%s' % futures.name
        return futures.name

    @classmethod
    def get_risk(cls):
        return RISK

    @classmethod
    def get_clevel(cls):
        return CLEVEL

    @classmethod
    def get_opcycle(cls):
        return OPCYCLE

    @classmethod
    def get_capital(cls):
        assert CAPITAL > 0
        return CAPITAL

    @classmethod
    def get_agency(cls):
        return AGENCY

    @classmethod
    def get_random(cls):
        from numpy import random
        return 1 if random.rand() <= RANDOM else 0

    @classmethod
    def get_stopbuy(cls):
        id = ID[-1]
        #print id[:2]
        #print int(id[2:])
        if int(id[2:]) == 1:
            m = '12'
            y = str(int(id[:2]) - 1)
        else:
            m = str(int(id[2:]) - 1)
            y = id[:2]
        d = 30 - STOPBUY
        if 0 < d < 10:
            d = '0' + str(d)
        elif d >= 10:
            d = str(d)
        else:
            raise AssertionError
        concat = '20' + y + '-' + m + '-' + d + ' 00:00:00'
        print concat
        return concat

    @classmethod
    def get_tt(cls):
        assert 0 < TT['IN'] <= 120
        assert 0 < TT['OUT'] <= 120
        assert 0 < TT['N'] <= 120
        assert isinstance(TT['IN'], int)
        assert isinstance(TT['OUT'], int)
        assert isinstance(TT['N'], int)
        return TT

    @classmethod
    def get_maxunit(cls):
        assert MAXUNIT > 0
        return MAXUNIT

    @classmethod
    def get_contractlist(cls):
        lst = []
        L = LIST(RANGE, ID)
        for l in L:
            lst.append(l.split('1')[0])
        return lst

    @classmethod
    def get_futures(cls):
        if PCON == 'AUTO':
            # from mul_base import run_mychoice
            allcontracts = LIST(RANGE, ID)
            print 'All contracts are: %s' % allcontracts
            futures = Futures(allcontracts)

            return futures
            assert 'ERROR pcon'

        else:
            raise AssertionError
            # return PCON

    @classmethod
    def get_restraint(cls):
        r = res()

        r.high = HIGHRIS
        r.low = LOWRIS
        r.other = OTHERS

        return r


class res(object):
    def __init__(self):
        self.high = dict()
        self.low = dict()
        self.other = dict()

'''Remove run_mychoice from V170608
Join in the get_name()
'''
'''
def run_mychoice():
    from mychoice import CsvSource, Choice
    # from mul_config import LIST, MAXCONTRACT

    c = CsvSource('./data/1DAY/SHFE')

    lst = LIST(RANGE, ID)
    ch = Choice()
    choice = []

    for contract in lst:
        pcon = contract + '.csv'
        df = c.get_tables(pcon)
        if ML:
            choice = ch.ml_selection(contract)
        else:
            choice = ch.selection(df, contract)

    # print yellow
    print '***  The selected contract is :  ***'
    print choice
    # print yellow_
    p = []
    for i in choice:
        if ML:
            p.append(i[1])
        else:
            p.append(i[2])
    p.reverse()
    print 'p=%s' % p

    if len(p) <= MAXCONTRACT:
        return p
    else:
        return p[0: MAXCONTRACT]
'''



class ML_Default(object):
    # Add machine learning para below!
    def __init__(self):
        pass

    @classmethod
    def get_close_idx(cls):
        return CLOSE_IDX

    @classmethod
    def get_high_idx(cls):
        return HIGH_IDX

    @classmethod
    def get_low_idx(cls):
        return LOW_IDX

    @classmethod
    def get_pred_range(cls):
        return PRED_RANGE

    @classmethod
    def get_split_ratio(cls):
        return SPLIT_RATIO

    @classmethod
    def get_num_steps(cls):
        return NUM_STEPS


class Futures(object):
    def __init__(self, lst):
        self.pcon = lst
        print 'pcon=%s' % self.pcon

        self.name = []
        for s in self.pcon:
            self.name.append(s + '.' + EXCHANGE + '-' + PERIOD[0] + '.' + PERIOD[1:])

        # @TODO Define what the pcon is look like
        '''status = {pcon: [activation, date, unit, count1day]}'''
        self.stat = {}
        for i in self.pcon:
            self.stat[i] = [0, '1999-01-01', 0, 0]
        print self.stat

    def activation(self):
        pass

    def status(self):
        pass


if __name__ == '__main__':
    # Default.name = '1'
    # print (Default.get_traindir().split('/')[2] + Default.get_traindir().split('/')[4].split('.')[0])
    # print Default.get_name().split('-')[1][2:]
    pass
