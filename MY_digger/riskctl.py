# -*- coding: utf-8 -*-
import csv


class Riskctl(object):

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

    #@classmethod
    def csv_test(self, code=None):
        n = []
        with open('./data/contracts.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[self.cts['code']] == code:
                    n.append(row)
        return n


if __name__ == '__main__':
    s = 'I.SHFE-1.Day'
    r = Riskctl()
    m = r.csv_test(s.split('.')[0])
    print m
    #assert (len(m) == 1)
