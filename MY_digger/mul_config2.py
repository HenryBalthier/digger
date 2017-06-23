# -*- coding: utf-8 -*-


# 关键参数
# PCON = 'AUTO'            # 合约简称
PCON = ['AG', 'B']
EXCHANGE = 'SHFE'
PERIOD = '1DAY'         # 合约数据时间格式
LIST = ['A', 'B', 'BB', 'I', 'AG']

# 回测相关参数
CAPITAL = 1000000       # 初始资金总数
RISK = 0.05             # 承担风险指数
CLEVEL = 0              # 机器学习预测置信比
OPCYCLE = 4             # 操作周期 = OPCYCLE * 数据最小时间单位

# 自定义策略相关参数
MAXUNIT = 4             # 最大头寸数限制
TT = {                  # 海龟相关数据（单位时间倍率）
    'IN': 20,           # 入市指标
    'OUT': 10,          # 退市指标
    'N': 20             # 波动率
}

# 机器学习相关参数
CLOSE_IDX = 1
HIGH_IDX = 2
LOW_IDX = 3
PRED_RANGE = 5
SPLIT_RATIO = 0.2
PCON_DICT = {'AG': '豆', 'B': '螺纹钢'}

# 态势分析相关参数
# 暂无


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


class Default(object):

    @classmethod
    def get_datadir(cls):
        pcon = cls.get_choice()
        data_dir = []
        for s in pcon:
            data_dir.append('./traindata/' + PERIOD + '/' + EXCHANGE + '/' + s + '_full.csv')
        return data_dir

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

    @classmethod
    def get_name(cls):
        pcon = cls.get_choice()
        name = []
        for s in pcon:
            name.append(s + '.' + EXCHANGE + '-' + PERIOD[0] + '.' + PERIOD[1:])
        return name

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
        return LIST

    @classmethod
    def get_choice(cls):
        if PCON == 'AUTO':
            assert 'ERROR pcon'

        else:
            return PCON


if __name__ == '__main__':
    # Default.name = '1'
    # print (Default.get_traindir().split('/')[2] + Default.get_traindir().split('/')[4].split('.')[0])
    # print Default.get_name().split('-')[1][2:]
    pass
