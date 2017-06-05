
class SplitFile(object):
    def __init__(self):
        self.fname = './traindata/1DAY/SHFE/A_full.csv'   # file name
        self.split_ratio = 0.2   # ratio for training data

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


if __name__ == '__main__':
    r = SplitFile()
    r.split()
