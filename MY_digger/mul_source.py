# -*- coding: utf-8 -*-

import csv
import pandas as pd
import os


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


def cut_idxs(prename, newname, idxs):
    with open(prename, 'r') as cvsin, open(newname, 'w') as cvsout:
        reader = csv.reader(cvsin)
        writer = csv.writer(cvsout)
        rows = (tuple(item for idx, item in enumerate(row) if idx not in idxs) for row in reader)
        writer.writerows(rows)


def cut_title(prename, newname):
    with open(prename, 'r') as cvsin, open(newname, 'w') as cvsout:
        reader = csv.reader(cvsin)
        writer = csv.writer(cvsout)
        i = 0
        rows = []
        for row in reader:
            if i != 0:
                rows.append(row)

            i += 1
        writer.writerows(rows)

def replace_title(prename, newname, title):
    with open(prename, 'r') as cvsin, open(newname, 'w') as cvsout:
        reader = csv.reader(cvsin)
        writer = csv.writer(cvsout)
        i = 0
        rows = []
        rows.append(title)
        for row in reader:
            if i != 0:
                rows.append(row)

            i += 1
        writer.writerows(rows)


def data_clip(prename, newname):
    a = pd.read_csv(prename)
    for list in ['No', 'preclose', 'predeal', 'keep', 'deal', 'up1', 'up2', 'cash']:
        del a[list]

    a['datetime'] = pd.to_datetime(a['datetime'], format='%Y%m%d', errors='coerce')
    print a['datetime']
    print type(a['datetime'])
    a['high'] /= 100.0
    a['low'] /= 100.0
    a['open'] /= 100.0
    a['close'] /= 100.0
    a['volume'] /= 1.0

    print (a['datetime'].index)
    a.to_csv(newname, index=False)

    print a['pcon']

    n = len(a['datetime'])
    while n > 0:
        fname = prename.split('1')[0] + a['pcon'][0].upper() + '.csv'
        #fname = a['pcon'][0] + '.csv'

        b = a
        for i in range(n):
            if a['pcon'][0] != b['pcon'][i]:
                b = b.drop(i)

        b = b.reset_index(drop=True)
        print b['pcon'][0]
        print a['pcon']
        for i in range(n):
            if a['pcon'][i] == b['pcon'][0]:
                a = a.drop(i)

        del b['pcon']
        a = a.reset_index(drop=True)
        n = len(a['datetime'])

        if os.path.exists(fname):
            print fname
            pre_a = pd.read_csv(fname)
            pre_a['datetime'] = pd.to_datetime(pre_a['datetime'], format='%Y-%m-%d')
            print pre_a['datetime']

            b = pd.concat([pre_a, b])
            print b['datetime']


        b.to_csv(fname, index=False)

        data_sort(fname)


def data_sort(fname):
    a = pd.read_csv(fname)
    n = len(a['datetime']) - 1

    a = a.reset_index(drop=True)
    a = a.sort_values(['datetime'])
    a = a.reset_index(drop=True)

    i = 0
    while i < n:
        j = i + 1
        while a['datetime'][i] == a['datetime'][j]:
            j += 1
            if j >= n:
                break
        lst = []
        if j != (i + 1):
            for l in range(i + 1, j):
                lst.append(l)

        for m in lst:
            a = a.drop(m)
        i = j

    a = a.reset_index(drop=True)
    a = a.sort_values(['datetime'])
    a = a.reset_index(drop=True)

    n = len(a['datetime'])
    for i in range(n):
        for l in ['open', 'high', 'low', 'close']:      # 'volume'
            if a[l][i] == 0.0:
                a = a.drop(i)
                break
    a.to_csv(fname, index=False)

'''
def date_devide(prename, newname, title):
    with open(prename, 'r') as cvsin, open(newname, 'w') as cvsout:
        reader = csv.reader(cvsin)
        writer = csv.writer(cvsout)
        i = 0
        rows = []
        a = None
        # rows.append(title)
        for row in reader:
            if i == 0:
                a = row[0]
                print a
                rows.append(row)
            elif i != 0 and row[0] == a:
                rows.append(row)

            i += 1
        writer.writerows(rows)
'''

if __name__ == '__main__':
    # s = 'I.SHFE-1.Day'
    # source = Source()
    # source.csv_test('./source/ag14.csv', './source/1.csv', [0])
    # source.pd_test('./source/B.csv', './source/2.csv', [0])
    # source.mainpcon('./source/B14.csv', './source/B.csv')
    # source.combine('a', 11, 15)

    f0 = './source/大连15/'

    f1 = './source/1.csv'
    f2 = './source/2.csv'
    title = ['No', 'pcon', 'datetime', 'preclose', 'predeal', 'open', 'high', 'low', 'close',
             'deal', 'up1', 'up2', 'volume', 'cash', 'keep']
    title2 = ['pcon', 'datetime', 'open', 'high', 'low', 'close', 'volume']

    #lst15 = ['A', 'B', 'BB', 'C', 'CS', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'M', 'P', 'PP', 'V', 'Y']
    lst15 = ['A']
    for i in lst15:
        f = f0 + i + '-2015.csv'
        replace_title(f, f1, title)
        data_clip(f1, f2)

    #cut_title(f0, f1)

    #cut_title(f2, f3)
    #date_devide(f3, f4, title2)
