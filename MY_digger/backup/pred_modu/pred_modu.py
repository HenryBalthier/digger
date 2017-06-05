# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
# from keras.layers.wrappers import TimeDistributed
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import Default, ML_Default
from utils import MergeData

traindir = Default.get_traindir()
sadir = Default.get_sadir()
reload(sys)
sys.setdefaultencoding('utf-8')


class Train(object):

    def __init__(self):
        np.random.seed(5)  # For reproducability
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
        df = MergeData(data_frame[1:], sa_frame[1:]).merge()
        self.dataset = df
        self.dataset = self.dataset.astype('float32')
        self.close_idx = ML_Default.get_close_idx()   # index of the column of close price
        self.num_steps = 30
        self.pred_range = ML_Default.get_pred_range()  # if 1, predict the next period close price
        self.scaler = []

    def _create_dataset(self, _dat, _num_steps=30, _pred_range=1):
        dat_x = []
        dat_y = []
        for _i in xrange(_dat.shape[0] - _pred_range):
            if _i < _num_steps:
                continue
            dat_x.append(_dat[_i - _num_steps: _i, :])
            tmp_i = _i + 1
            dat_max = np.max(_dat[tmp_i:(tmp_i + _pred_range), self.close_idx], axis=0)
            dat_min = np.min(_dat[tmp_i:(tmp_i + _pred_range), self.close_idx], axis=0)
            _tmp = _dat[_i, self.close_idx]
            if np.abs(dat_max - _tmp) > np.abs(dat_min - _tmp):
                dat_y.append(dat_max)
            else:
                dat_y.append(dat_min)
        return np.array(dat_x), np.array(dat_y)

    def run(self):

        train_data = self.dataset[:, :]
        feature_dim = train_data.shape[1]
        # normalize the dataset
        for i in xrange(feature_dim):
            self.scaler.append(StandardScaler().fit(train_data[:, i].reshape(-1, 1)))
            train_data[:, i] = self.scaler[i].transform(train_data[:, i].reshape(-1, 1)).reshape(-1,)
        train_x, train_y = self._create_dataset(train_data, self.num_steps, self.pred_range)
        target_dim = 1
        # reshape input and output to be [samples, time steps, features]
        train_x = train_x.reshape((train_x.shape[0], self.num_steps, feature_dim))
        train_y = train_y.reshape((train_y.shape[0], target_dim))
        print("Data loaded.")

        in_neurons = feature_dim
        out_neurons = target_dim
        hidden_neurons = 100
        batch_size = 100
        num_epoch = 50
        model = Sequential()
        # model.add(LSTM(hidden_neurons, input_dim=in_neurons, return_sequences=True))
        # model.add(TimeDistributed(Dense(out_neurons, activation="linear")))
        model.add(LSTM(hidden_neurons, input_dim=in_neurons, return_sequences=False))
        model.add(Dense(out_neurons, activation="linear"))
        model.compile(loss="mean_absolute_percentage_error", optimizer="adam",
                      metrics=["mean_absolute_percentage_error"])
        print("Model compiled.")
        model.fit(train_x, train_y, batch_size=batch_size, validation_split=0.2,
                  nb_epoch=num_epoch, verbose=2)
        print("Model trained.")
        filename = traindir.split('/')[2] + '_' + traindir.split('/')[4].split('.')[0] + '.h5'
        model.save(filename)
        print("Model saved.")

if __name__ == '__main__':
    t = Train()
    t.run()
