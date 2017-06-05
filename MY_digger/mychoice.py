# -*- coding: utf-8 -*-

import os
import pandas as pd


class CsvSource(object):
    """ Date Source"""

    def __init__(self, root):
        self._root = root

    def get_code2strpcon(self):
        symbols = {}  # code -> string pcontracts
        period_exchange2strpcon = {}  # exchange.period -> string pcontracts
        for parent, dirs, files in os.walk(self._root):
            if dirs == []:
                t = parent.split(os.sep)
                period, exch = t[-2], t[-1]
                for i, a in enumerate(period):
                    if not a.isdigit():
                        sepi = i
                        break
                count = period[0:sepi]
                unit = period[sepi:]
                period = '.'.join([count, unit])
                strpcons = period_exchange2strpcon.setdefault(
                    ''.join([exch, '-', period]), [])
                for file_ in files:
                    if file_.endswith('csv'):
                        code = file_.split('.')[0]
                        t = symbols.setdefault(code, [])
                        rst = ''.join([code, '.', exch, '-', period])
                        t.append(rst)
                        strpcons.append(rst)
        return symbols, period_exchange2strpcon

    def import_contracts(self, data):
        """ 导入合约的基本信息。

        Args:
            data (dict): {key, code, exchange, name, spell,
            long_margin_ratio, short_margin_ratio, price_tick, volume_multiple}

        """
        fname = os.path.join(self._root, "contracts.csv")
        df = pd.DataFrame(data)
        df.to_csv(fname, columns=[
            'code', 'exchange', 'name', 'spell',
            'long_margin_ratio', 'short_margin_ratio', 'price_tick',
            'volume_multiple'
        ], index=False)

    def import_bars(self, tbdata, pcontract):
        """ 导入交易数据

        Args:
            tbdata (dict): {'datetime', 'open', 'close',
                            'high', 'low', 'volume'}
            pcontract (PContract): 周期合约
        """
        strpcon = str(pcontract).upper()
        contract, period = tuple(strpcon.split('-'))
        code, exch = tuple(contract.split('.'))
        period = period.replace('.', '')
        try:
            os.makedirs(os.path.join(self._root, period, exch))
        except OSError:
            pass
        fname = os.path.join(self._root, period, exch, code+'.csv')
        df = pd.DataFrame(tbdata)
        df.to_csv(fname, columns=[
            'datetime', 'open', 'close', 'high', 'low', 'volume'
        ], index=False)

    def get_contracts(self):
        """ 获取所有合约的基本信息

        Returns:
            pd.DataFrame
        """
        fname = os.path.join(self._root, "contracts.csv")
        df = pd.read_csv(fname)
        df.index = df['code'] + '.' + df['exchange']
        df.index = map(lambda x: x.upper(), df.index)
        return df

    def export_bars(self, index=True, index_label='index'):
        """
            导出csv中的所有表格数据。
        """
        pass

    def get_tables(self, name=None):
        """ 返回数据库所有的表格"""
        fname = os.path.join(self._root, name)
        df = pd.read_csv(fname)
        return df


class Choice(object):

    def __init__(self):
        self.flow = 0
        self.Max = 0
        self.Min = 0
        self.d = -1
        self.lst = []

    def on_init(self):
        self.flow = 0
        self.Max = 0
        self.Min = 0
        self.d = -1

    def selection(self, df, pcon):
        l = len(df)
        self.on_init()

        if l > 100:
            # print (df[l - 100: l])
            for i in range(l - 100, l):
                self.flow += df["volume"][i]
                self.d = 100
                if self.Max < df["close"][i]:
                    self.Max = df["close"][i]
                if self.Min > df["close"][i]:
                    self.Min = df["close"][i]
        else:
            # print (df[1: l])
            for i in range(1, l):
                self.flow += df["volume"][i]
                self.d = l
                if self.Max < df["close"][i]:
                    self.Max = df["close"][i]
                if self.Min > df["close"][i]:
                    self.Min = df["close"][i]

        n = round((self.Max - self.Min), 2)
        avg = round((self.flow / self.d), 2)

        print ("In last %d units" % self.d)
        print ("average of the VOLUME: %.2f" % avg)     # 近100days的平均交易量
        print ("max of the N: %.2f" % n)                # 近100days的波动率

        return self.sort_push(pcon)

    def sort_push(self, pcon):
        n = round((self.Max - self.Min), 2)
        avg = round((self.flow / self.d), 2)
        lst = [n, avg, pcon]
        self.lst.append(lst)

        """
        x[0]: N first
        x[1]: Volume first
        """

        l = sorted(self.lst, key=lambda x: x[1])
        print l
        return l[-1]


if __name__ == '__main__':
    c = CsvSource('./data/1DAY/SHFE')
    lst = ['A', 'B', 'BB', 'I']
    ch = Choice()
    choice = []

    for contract in lst:
        pcon = contract + '.csv'
        df = c.get_tables(pcon)
        choice = ch.selection(df, contract)

    yellow = '\033[1;33m'
    yellow_ = '\033[0m'
    print yellow
    print '***  The selected contract is :  ***'
    print choice
    print yellow_
