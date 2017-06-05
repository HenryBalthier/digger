# -*- coding: utf-8 -*-
import sys
from keras.models import load_model
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from MY_digger.config import Default, ML_Default
from utils import MergeData


"""
This module should be invoked once when a new data comes in.
"""

testdir = Default.get_testdir()
sadir = Default.get_sadir()
reload(sys)
sys.setdefaultencoding('utf-8')


class Predict(object):
    # prepare data
    # this should be replaced by new price data of the same future.
    # expecting: dat_input = [open, close, high, low, volume]

    def __init__(self):

        data_frame = []
        with open(testdir, 'r') as data_file:
            reader = csv.reader(data_file)
            for line in reader:
                data_frame.append(line)
        sa_frame = []
        with open(sadir, 'r') as sa_file:
            reader = csv.reader(sa_file)
            for line in reader:
                sa_frame.append(line)
        df = MergeData(data_frame[1:], sa_frame[1:]).merge()
        self.dat_input = df
        self.dat_input = self.dat_input.astype('float32')
        self.dat_output = None
        self.confidence = None
        self.confidence_lookback = 5
        self.close_idx = ML_Default.get_close_idx()
        self.pred_range = ML_Default.get_pred_range()
        self.scaler = []

    def run(self):
        # normalize the dataset
        for i in xrange(5):
            self.scaler.append(StandardScaler().fit(self.dat_input[:, i].reshape(-1, 1)))
            self.dat_input[:, i] = self.scaler[i].transform(self.dat_input[:, i].reshape(-1, 1)).reshape(-1,)
        print "Data loaded."

        # load model
        filename = testdir.split('/')[2] + '_' + testdir.split('/')[4].split('.')[0] + '.h5'
        model = load_model(filename)
        print "Model loaded."

        feature_dim = self.dat_input.shape[1]
        model_input = self.dat_input.reshape((self.dat_input.shape[0], 1, feature_dim))

        dat_predict = []
        dat_mape = []    # mean_absolute_percentage_error
        for i in xrange(self.confidence_lookback):
            dat_mape.append(100)    # for confidence generation
        for i in xrange(model_input.shape[0]):
            dat_x = model_input[i, :, :].reshape(1, 1, feature_dim)
            dat_predict.append(model.predict_on_batch(dat_x))
            if i <= model_input.shape[0] - self.pred_range:
                tmp_i = i + 1
                dat_max = np.max(model_input[tmp_i:(tmp_i + self.pred_range), :, self.close_idx], axis=0)
                dat_min = np.min(model_input[tmp_i:(tmp_i + self.pred_range), :, self.close_idx], axis=0)
                _tmp = model_input[i, :, self.close_idx]
                if np.abs(dat_max - _tmp) > np.abs(dat_min - _tmp):
                    dat_y = dat_max
                else:
                    dat_y = dat_min
                _, mape = model.test_on_batch(dat_x, dat_y)
                dat_mape.append(mape)
                model.train_on_batch(dat_x, dat_y)
            else:
                dat_mape.append(100)

        dat_predict = np.array(dat_predict).reshape(-1,)
        close_price = self.dat_input[:, self.close_idx]
        self.dat_output = (dat_predict - close_price) / close_price
        dat_mape = 100 - np.array(dat_mape)
        confidence_tmp = []
        for i in xrange(dat_predict.shape[0]):
            confidence_tmp.append(np.mean(dat_mape[i: i + self.confidence_lookback]))
        self.confidence = np.array(confidence_tmp)
        #print(np.dstack((self.dat_output, self.confidence)))
        #print(self.confidence)

        # plot baseline and predictions
        if __name__ == '__main__':
            import matplotlib.pyplot as plt
            plt.plot(close_price, 'b')
            plt.plot(dat_predict, 'r')
            plt.show()

if __name__ == '__main__':
    p = Predict()
    p.run()
