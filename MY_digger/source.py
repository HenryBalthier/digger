# -*- coding: utf-8 -*-

import csv
import pandas as pd


class Source(object):

    def csv_test(self, prename, newname, idxs):
        with open(prename, 'r') as cvsin, open(newname, 'w') as cvsout:
            reader = csv.reader(cvsin)
            writer = csv.writer(cvsout)

            rows = (tuple(item for idx, item in enumerate(row) if idx not in idxs) for row in reader)
            writer.writerows(rows)

    def pd_test(self, prename, newname, idxs):
        a = pd.read_csv(prename)
        for list in ['No', 'pcon', 'preclose', 'predeal', 'keep', 'deal', 'up1', 'up2', 'cash']:
            del a[list]

        a['datetime'] = pd.to_datetime(a['datetime'], format='%Y%m%d', errors='coerce')

        n = len(a['datetime'])
        a = a.sort_values(['datetime'])
        a = a.reset_index(drop=True)
        for i in range(n):
            if (a['open'][i] + a['high'][i] + a['low'][i]) == 0:
                a = a.drop(i)

        a = a.reset_index(drop=True)
        n = len(a['datetime'])
        i = 0
        while i < n:
            j = i+1
            while a['datetime'][i] == a['datetime'][j]:
                j += 1
                if j >= n:
                    break
            lst = []
            if j != (i+1):
                for l in range(i, j):
                    for m in range(i, j):
                        if m != l:
                            if a['volume'][m] < a['volume'][l]:
                                if m not in lst:
                                    lst.append(m)
                            else:
                                if l not in lst:
                                    lst.append(l)

            for m in lst:
                a = a.drop(m)
            i = j
            print lst

        a['high'] = a['high'] / 100.0
        a['low'] = a['low'] / 100.0
        a['open'] = a['open'] / 100.0
        a['close'] = a['close'] / 100.0
        a['volume'] = a['volume'] / 1.0

        print (a['datetime'].index)
        a.to_csv(newname, index=False)

    def mainpcon(self, prename, newname):
        a = pd.read_csv(prename)
        for list in [
                     'No',
                     'preclose',
                     'predeal',
                     'keep',
                     'deal',
                     'up1',
                     'up2',
                     'cash'
                     ]:
            del a[list]

        a = a.fillna(method='pad')
        print a

        a['datetime'] = pd.to_datetime(a['datetime'], format='%Y%m%d', errors='coerce')
        n = len(a['datetime'])
        for i in range(n):
            if a['pcon'][i] == 0:
                a['pcon'][i] = a['pcon'][i-1]

        print a

        a = a.sort_values(['datetime'])
        a = a.reset_index(drop=True)
        for i in range(n):
            if (a['open'][i] + a['high'][i] + a['low'][i]) == 0:
                a = a.drop(i)

        a = a.sort_values(['pcon'])
        a = a.reset_index(drop=True)

        n = a.groupby(by=['pcon'])['volume'].sum()
        pcon = n.sort_values().index[-1]

        n = len(a['datetime'])
        a = a.sort_values(['datetime'])
        a = a.reset_index(drop=True)
        for i in range(n):
            if a['pcon'][i] != pcon:
                a = a.drop(i)

        for list in ['pcon']:
            del a[list]

        a['high'] = a['high'] / 100.0
        a['low'] = a['low'] / 100.0
        a['open'] = a['open'] / 100.0
        a['close'] = a['close'] / 100.0
        a['volume'] = a['volume'] / 1.0

        a.to_csv(newname, index=False)

    def combine(self, fname, start, end):
        n = end - start + 1
        fullname = './source/' + fname.upper() + '_full.csv'
        s = start
        with open(fullname, 'w') as f:
            for i in range(n):
                filename = './source/' + fname + str(s) + '.csv'
                print filename
                with open(filename, 'r') as a:
                    lines = a.readlines()
                    if s == start:
                        f.writelines(lines[0])
                    f.writelines(lines[1:])
                s += 1
            '''
            with open('./source/a11.csv', 'r') as a:
                lines = a.readlines()
                f.writelines(lines[])
            with open('./source/a12.csv', 'r') as a:
                lines = a.readlines()
                f.writelines(lines[1:])
            with open('./source/a13.csv', 'r') as a:
                lines = a.readlines()
                f.writelines(lines[1:])
            with open('./source/a14.csv', 'r') as a:
                lines = a.readlines()
                f.writelines(lines[1:])
            with open('./source/a15.csv', 'r') as a:
                lines = a.readlines()
                f.writelines(lines[1:])
            '''


if __name__ == '__main__':
    # s = 'I.SHFE-1.Day'
    source = Source()
    # source.csv_test('./source/ag14.csv', './source/1.csv', [0])
    # source.pd_test('./source/B.csv', './source/2.csv', [0])
    source.mainpcon('./source/B14.csv', './source/B.csv')
    # source.combine('a', 11, 15)
