from config import Default, ML_Default
import pandas as pd
import csv

traindir = Default.get_traindir()
sadir = Default.get_sadir()


class SplitFile(object):
    def __init__(self):
        self.fname = './traindata/1MINUTE/SHFE/BB_full.csv'   # file name
        self.split_ratio = ML_Default.get_split_ratio()   # ratio for training data

    def split(self):
        with open(self.fname) as f:
            for length, _ in enumerate(f):
                pass
        train_filesize = int(length * self.split_ratio)
        train_filename = self.fname[:-9] + '.train.csv'
        test_filename = self.fname[:-9] + '.csv'
        with open(self.fname) as f:
            lines = f.readlines()
            with open(train_filename, 'w') as train_file:
                train_file.writelines(lines[:train_filesize])
            with open(test_filename, 'w') as test_file:
                test_file.writelines(lines[0])
                test_file.writelines(lines[train_filesize:])


class MergeData(object):
    def __init__(self, f_data, sa_data):
        self.f_data = pd.DataFrame(f_data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        self.sa_data = pd.DataFrame(sa_data, columns=['datetime', 'sa'])

    def merge(self):
        df = pd.merge(self.f_data, self.sa_data, how='left', on='datetime')
        df = df.fillna('0')
        return df.values[:, 1:]


if __name__ == '__main__':
    # r = SplitFile()
    # r.split()
    data_frame = []
    with open(traindir, 'r') as data_file:
        reader = csv.reader(data_file)
        for line in reader:
            data_frame.append(line)
    sa_frame = []
    with open(sadir, 'r') as sa_file:
        reader = csv.reader(sa_file)
        for line in reader:
            sa_frame.append(line)
    d = MergeData(data_frame[1:], sa_frame[1:]).merge()
    print(d)
