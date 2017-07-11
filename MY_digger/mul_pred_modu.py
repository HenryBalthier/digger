# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.layers.wrappers import TimeDistributed
from sklearn.preprocessing import StandardScaler
import mul_config
from mul_config import ML_Default
import MySQLdb
import urllib2
import re
import datetime
import gensim
import jieba
import pickle
import string
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


class Utilbox(object):

    @classmethod
    def dat_for_price_pred(cls, _dat, _num_steps=30, _pred_range=5):
        """
        prepare for batch prediction.
        :param _dat:
        :param _num_steps: number of steps to look back
        :param _pred_range:
        :return:
        """
        close_idx = ML_Default.get_close_idx()
        dat_x = []
        dat_y = []

        for _i in xrange(_dat.shape[0] - _pred_range + 1):
            if _i < _num_steps:
                continue
            dat_x.append(_dat[_i - _num_steps: _i, :])
            dat_max = np.max(_dat[_i:(_i + _pred_range), close_idx], axis=0)
            dat_min = np.min(_dat[_i:(_i + _pred_range), close_idx], axis=0)
            dat_y.append([dat_max, dat_min])
        return np.array(dat_x), np.array(dat_y)

    @classmethod
    def dat_for_price_seq_pred(cls, _dat, _num_steps=30, _pred_range=5):
        """
        prepare for sequantial prediction
        :param _dat:
        :param _num_steps: number of steps for truncated BPTT
        :param _pred_range:
        :return:
        """
        close_idx = ML_Default.get_close_idx()
        dat_x = []
        dat_y = []
        dat_y_seq = []
        for _i in xrange(_dat.shape[0] - _pred_range + 1):
            dat_max = np.max(_dat[_i:(_i + _pred_range), close_idx], axis=0)
            dat_min = np.min(_dat[_i:(_i + _pred_range), close_idx], axis=0)
            dat_y_seq.append([dat_max, dat_min])

        for _i in xrange(_dat.shape[0] - _pred_range + 1):
            if _i < _num_steps:
                continue
            dat_x.append(_dat[_i - _num_steps: _i, :])
            dat_y.append(dat_y_seq[_i - _num_steps: _i])
        return np.array(dat_x), np.array(dat_y), np.array(dat_y_seq)

    @classmethod
    def merge(cls, _f_data, _sa_data):
        f_data = pd.DataFrame(_f_data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        __sa_data = pd.DataFrame(_sa_data, columns=['datetime', 'sa'])
        # print(__sa_data)
        __sa_data['sa'] = pd.to_numeric(__sa_data['sa'])
        __sa_grouped = __sa_data.groupby('datetime', as_index=False)
        sa_data = pd.DataFrame(__sa_grouped.mean())
        # print(self.sa_data)
        df = pd.merge(f_data, sa_data, how='left', on='datetime')
        df = df.fillna('0')
        return df.values[:, 1:]

    @classmethod
    def split(cls, _s):
        s = _s
        fname = './data/' + mul_config.PERIOD + '/' + mul_config.EXCHANGE + '/' + s + '.csv'  # file name
        split_ratio = ML_Default.get_split_ratio()  # ratio for training data
        with open(fname) as f:
            for length, _ in enumerate(f):
                pass
        train_filesize = int(length * split_ratio)
        train_filename = './traindata/' + mul_config.PERIOD + '/' \
                         + mul_config.EXCHANGE + '/' + s + '.train.csv'   # file name
        with open(fname) as f:
            lines = f.readlines()
            with open(train_filename, 'w') as train_file:
                train_file.writelines(lines[:train_filesize])


class Train(object):

    def __init__(self, _s):
        data_frame = []
        self.s = _s
        self.train_dir = './traindata/' + mul_config.PERIOD + '/' \
                         + mul_config.EXCHANGE + '/' + self.s + '.train.csv'
        if not os.path.exists(self.train_dir):
            Utilbox.split(self.s)
        _sa_dir = './traindata/' + mul_config.PERIOD + '/' + mul_config.EXCHANGE + '/' + self.s + '.sa.csv'
        with open(self.train_dir, 'r') as data_file:
            reader = csv.reader(data_file)
            for line in reader:
                data_frame.append(line)
        sa_frame = []
        if not os.path.exists(_sa_dir):
            SAforTest.prepare_sa(self.s)
        with open(_sa_dir, 'r') as sa_file:
            reader = csv.reader(sa_file)
            for line in reader:
                sa_frame.append(line)
        df = Utilbox.merge(data_frame[1:], sa_frame[1:])
        self.dataset = df
        self.dataset = self.dataset.astype('float32')
        self.close_idx = ML_Default.get_close_idx()   # index of the column of close price
        self.high_idx = ML_Default.get_high_idx()
        self.low_idx = ML_Default.get_low_idx()
        self.num_steps = ML_Default.get_num_steps()
        self.pred_range = ML_Default.get_pred_range()  # if 1, predict the next period close price
        self.fluq_n = 0
        self.train_data = self.dataset[:, :]
        # normalize the dataset
        for i in xrange(self.train_data.shape[1]):
            scaler = StandardScaler().fit(self.train_data[:, i].reshape(-1, 1))
            self.train_data[:, i] = scaler.transform(self.train_data[:, i].reshape(-1, 1)).reshape(-1, )

    def _dat_for_fluq_pred(self, _dat, _num_steps=30):
        dat_x = []
        dat_y = []
        dat_fluq = []

        for _i in xrange(_dat.shape[0]):
            if _i == 0:
                dat_fluq.append(_dat[_i, self.high_idx] - _dat[_i, self.low_idx])
            else:
                dat_fluq.append(max(_dat[_i, self.high_idx] - _dat[_i, self.low_idx],
                                    _dat[_i, self.high_idx] - _dat[_i - 1, self.close_idx],
                                    _dat[_i - 1, self.close_idx] - _dat[_i, self.low_idx]))

        dat_fluq = np.array(dat_fluq)
        _len = dat_fluq.shape[0]

        # normalization
        _scaler_fluq = StandardScaler().fit(dat_fluq.reshape(-1, 1))
        dat_fluq = _scaler_fluq.transform(dat_fluq.reshape(-1, 1)).reshape(-1,)

        for _i in xrange(_len - 1):
            if _i < _num_steps:
                continue
            dat_x.append(dat_fluq[_i - _num_steps: _i])
            dat_y.append(dat_fluq[_i + 1])

        return np.array(dat_x), np.array(dat_y), np.array(dat_fluq[_len - _num_steps: _len]), _scaler_fluq

    def _cal_fluq_n(self):
        train_x, train_y, pred_x, scaler_fluq = self._dat_for_fluq_pred(self.train_data, self.num_steps)
        feature_dim = 1
        target_dim = 1
        # reshape input and output to be [samples, time steps, features]
        train_x = train_x.reshape((train_x.shape[0], self.num_steps, feature_dim))
        train_y = train_y.reshape((train_y.shape[0], target_dim))
        hidden_neurons = 32
        batch_size = 16
        num_epoch = 100
        model = Sequential()
        model.add(LSTM(hidden_neurons, input_dim=feature_dim, return_sequences=False))
        model.add(Dense(hidden_neurons, activation="tanh"))
        model.add(Dense(target_dim, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer="adam",
                      metrics=["mean_absolute_percentage_error"])
        model.fit(train_x, train_y, batch_size=batch_size, validation_split=0.2,
                  nb_epoch=num_epoch, verbose=2)
        pred_x = pred_x.reshape((1, self.num_steps, feature_dim))
        _fluq_n = model.predict_on_batch(pred_x)
        _fluq_n = scaler_fluq.inverse_transform(_fluq_n)
        return _fluq_n[0, 0]

    def run_fluq_n(self):
        self.run()
        return self._cal_fluq_n()

    def run(self):

        feature_dim = self.train_data.shape[1]
        train_x, train_y = Utilbox.dat_for_price_pred(self.train_data, self.num_steps, self.pred_range)
        target_dim = 2
        # reshape input and output to be [samples, time steps, features]
        train_x = train_x.reshape((train_x.shape[0], self.num_steps, feature_dim))
        train_y = train_y.reshape((train_y.shape[0], target_dim))
        print("Data loaded.")

        in_neurons = feature_dim
        out_neurons = target_dim
        hidden_neurons = 32
        batch_size = 4
        num_epoch = 10      #@TODO
        model = Sequential()
        # model.add(LSTM(hidden_neurons, input_dim=in_neurons, return_sequences=True))
        # model.add(TimeDistributed(Dense(out_neurons, activation="linear")))
        model.add(LSTM(hidden_neurons, input_dim=in_neurons, return_sequences=False))
        model.add(Dense(hidden_neurons, activation="tanh"))
        model.add(Dense(out_neurons, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_percentage_error"])
        print("Model compiled.")
        model.fit(train_x, train_y, batch_size=batch_size, validation_split=0.2, nb_epoch=num_epoch, verbose=2)
        print("Model trained.")
        filename = mul_config.PERIOD + '_' + self.s + '.h5'
        model.save(filename)
        print("Model saved.")

    def run_seq(self):
        feature_dim = self.train_data.shape[1]
        train_x, train_y, _ = Utilbox.dat_for_price_seq_pred(self.train_data, self.num_steps, self.pred_range)
        target_dim = 2
        print(train_x.shape)
        print(train_y.shape)
        # reshape input and output to be [samples, time steps, features]
        train_x = train_x.reshape((train_x.shape[0], self.num_steps, feature_dim))
        train_y = train_y.reshape((train_y.shape[0], self.num_steps, target_dim))
        print("Data loaded.")

        in_neurons = feature_dim
        out_neurons = target_dim
        hidden_neurons = 32
        batch_size = 4
        num_epoch = 300
        model = Sequential()
        model.add(LSTM(hidden_neurons, input_dim=in_neurons, return_sequences=True))
        model.add(TimeDistributed(Dense(hidden_neurons, activation="tanh")))
        model.add(TimeDistributed(Dense(out_neurons, activation="linear")))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_percentage_error"])
        print("Model compiled.")
        model.fit(train_x, train_y, batch_size=batch_size, validation_split=0.2, nb_epoch=num_epoch, verbose=2)
        print("Model trained.")
        filename = mul_config.PERIOD + '_' + self.s + '.h5'
        model.save(filename)
        print("Model saved.")


class Predict(object):

    def __init__(self, _s_list):
        """
        :param _s_list: list of future number
        """

        self.s_list = _s_list
        self.dat_output = {}
        self.confidence = {}

    def run(self):
        for _s in self.s_list:
            data_frame = []
            test_dir = './traindata/' + mul_config.PERIOD + '/' + mul_config.EXCHANGE + '/' + _s + '.csv'

            _sa_dir = './traindata/' + mul_config.PERIOD + '/' + mul_config.EXCHANGE + '/' + _s + '.sa.csv'
            with open(test_dir, 'r') as data_file:
                reader = csv.reader(data_file)
                for line in reader:
                    data_frame.append(line)
            sa_frame = []
            with open(_sa_dir, 'r') as sa_file:
                reader = csv.reader(sa_file)
                for line in reader:
                    sa_frame.append(line)
            df = Utilbox.merge(data_frame[1:], sa_frame[1:])
            dat_input = df
            dat_input = dat_input.astype('float32')
            confidence_lookback = 5
            close_idx = ML_Default.get_close_idx()
            pred_range = ML_Default.get_pred_range()
            num_steps = ML_Default.get_num_steps()
            # normalize the dataset
            for i in xrange(5):
                scaler = StandardScaler().fit(dat_input[:, i].reshape(-1, 1))
                dat_input[:, i] = scaler.transform(dat_input[:, i].reshape(-1, 1)).reshape(-1,)
            print("Data loaded.")

            # load model
            filename = mul_config.PERIOD + '_' + _s + '.h5'
            model = load_model(filename)
            print("Model loaded.")

            model_input, model_output = Utilbox.dat_for_price_pred(dat_input, num_steps, pred_range)

            dat_predict = []
            max_mape = []    # mean_absolute_percentage_error for predicted "max value"
            min_mape = []

            for i in xrange(num_steps - 1):
                max_mape.append(100)  # for confidence generation
                min_mape.append(100)
                dat_predict.append([0, 0])
            for i in xrange(model_input.shape[0]):
                dat_x = np.array(model_input[i, :, :]).reshape((1, model_input.shape[1], model_input.shape[2]))
                dat_y = np.array(model_output[i, :]).reshape(1, model_output.shape[1])
                dat_predict_tmp = model.predict_on_batch(dat_x)
                cp_tmp = dat_x[0, -1, close_idx]
                if dat_y[0, 0] - cp_tmp == 0:
                    max_mape_tmp = 100
                else:
                    max_mape_tmp = np.abs(dat_predict_tmp[0, 0] - dat_y[0, 0] / dat_y[0, 0] - cp_tmp) * 100
                if dat_y[0, 1] - cp_tmp == 0:
                    min_mape_tmp = 100
                else:
                    min_mape_tmp = np.abs(dat_predict_tmp[0, 1] - dat_y[0, 1] / dat_y[0, 1] - cp_tmp) * 100
                dat_predict.append(dat_predict_tmp[0, :])
                max_mape.append(max_mape_tmp)
                min_mape.append(min_mape_tmp)
                model.train_on_batch(dat_x, dat_y)
            for i in xrange(pred_range):
                max_mape.append(100)  # for confidence generation
                min_mape.append(100)
                dat_predict.append([0, 0])

            dat_predict = np.array(dat_predict)
            max_mape = 100 - np.array(max_mape)
            min_mape = 100 - np.array(min_mape)
            max_confidence_tmp = []
            min_confidence_tmp = []
            confidence_tmp = []
            _dat_output = []
            x_axis = []
            x_axis_value = 0.0
            for i in xrange(confidence_lookback):
                x_axis.append(x_axis_value)
                x_axis_value += 1.0
            x_axis = np.array(x_axis)
            for i in xrange(confidence_lookback):
                max_confidence_tmp.append(0)
                min_confidence_tmp.append(0)
            # least squares of polynomial fit of confience value
            for i in xrange(dat_predict.shape[0] - confidence_lookback):
                y_axis = max_mape[i: i + confidence_lookback]
                z = np.polyfit(x_axis, y_axis, 3)
                p = np.poly1d(z)
                max_confidence_tmp.append(p(x_axis_value))
                y_axis = min_mape[i: i + confidence_lookback]
                z = np.polyfit(x_axis, y_axis, 3)
                p = np.poly1d(z)
                min_confidence_tmp.append(p(x_axis_value))

            for i in xrange(dat_predict.shape[0]):
                cp_tmp = dat_input[i, close_idx]
                if max_confidence_tmp[i] > min_confidence_tmp[i]:
                    confidence_tmp.append(max_confidence_tmp[i])
                    _dat_output.append((dat_predict[i, 0] - cp_tmp) / cp_tmp)
                else:
                    confidence_tmp.append(min_confidence_tmp[i])
                    _dat_output.append((dat_predict[i, 1] - cp_tmp) / cp_tmp)
            self.confidence[_s] = np.array(confidence_tmp)
            self.dat_output[_s] = np.array(_dat_output)
            # print(np.dstack((self.dat_output, self.confidence)))
            # plot baseline and predictions
            plt.plot(self.dat_output[_s], 'r')
            plt.show()
        return self.dat_output, self.confidence

    def run_seq(self):
        """
        This function would be used on real trading situration.
        :return:
        """
        for _s in self.s_list:
            data_frame = []
            test_dir = './traindata/' + mul_config.PERIOD + '/' + mul_config.EXCHANGE + '/' + _s + '.csv'
            _sa_dir = './traindata/' + mul_config.PERIOD + '/' + mul_config.EXCHANGE + '/' + _s + '.sa.csv'
            with open(test_dir, 'r') as data_file:
                reader = csv.reader(data_file)
                for line in reader:
                    data_frame.append(line)
            sa_frame = []
            with open(_sa_dir, 'r') as sa_file:
                reader = csv.reader(sa_file)
                for line in reader:
                    sa_frame.append(line)
            df = Utilbox.merge(data_frame[1:], sa_frame[1:])
            close_idx = ML_Default.get_close_idx()
            pred_range = ML_Default.get_pred_range()
            num_steps = ML_Default.get_num_steps()
            dat_input = df[:-pred_range, :]
            dat_input = dat_input.astype('float32')
            # normalize the dataset
            for i in xrange(dat_input.shape[1]):
                scaler = StandardScaler().fit(dat_input[:, i].reshape(-1, 1))
                dat_input[:, i] = scaler.transform(dat_input[:, i].reshape(-1, 1)).reshape(-1, )
            print("Data loaded.")

            # load model
            filename = mul_config.PERIOD + '_' + _s + '.h5'
            model = load_model(filename)
            print("Model loaded.")

            model_input = dat_input.reshape(1, dat_input.shape[0], dat_input.shape[1])
            _, _, model_output = Utilbox.dat_for_price_seq_pred(dat_input, num_steps, pred_range)
            dat_predict = model.predict(model_input)
            dat_predict = np.array(dat_predict).reshape(-1, 2)   # 3 dimensions to 2 dimensions for convenience
            confidence_lookback = 5

            max_mape = []  # mean_absolute_percentage_error for predicted "max value"
            min_mape = []

            for i in xrange(model_input.shape[1]- pred_range):
                cp_tmp = dat_input[i, close_idx]
                if model_output[i, 0] - cp_tmp == 0:
                    max_mape_tmp = 100
                else:
                    max_mape_tmp = np.abs(dat_predict[i, 0] - model_output[i, 0] / model_output[i, 0] - cp_tmp) * 100
                if model_output[i, 1] - cp_tmp == 0:
                    min_mape_tmp = 100
                else:
                    min_mape_tmp = np.abs(dat_predict[i, 1] - model_output[i, 1] / model_output[i, 1] - cp_tmp) * 100
                max_mape.append(max_mape_tmp)
                min_mape.append(min_mape_tmp)
                # model.train_on_batch(dat_x, dat_y)
            for i in xrange(pred_range):
                max_mape.append(100)  # for confidence generation
                min_mape.append(100)

            max_mape = 100 - np.array(max_mape)
            min_mape = 100 - np.array(min_mape)
            max_confidence_tmp = []
            min_confidence_tmp = []
            confidence_tmp = []
            _dat_output = []
            x_axis = []
            x_axis_value = 0.0
            for i in xrange(confidence_lookback):
                x_axis.append(x_axis_value)
                x_axis_value += 1.0
            x_axis = np.array(x_axis)
            for i in xrange(confidence_lookback):
                max_confidence_tmp.append(0)
                min_confidence_tmp.append(0)
            # least squares of polynomial fit of confience value
            for i in xrange(dat_predict.shape[0] - confidence_lookback):
                y_axis = max_mape[i: i + confidence_lookback]
                z = np.polyfit(x_axis, y_axis, 3)
                p = np.poly1d(z)
                max_confidence_tmp.append(p(x_axis_value))
                y_axis = min_mape[i: i + confidence_lookback]
                z = np.polyfit(x_axis, y_axis, 3)
                p = np.poly1d(z)
                min_confidence_tmp.append(p(x_axis_value))

            for i in xrange(dat_predict.shape[0]):
                cp_tmp = dat_input[i, close_idx]
                if max_confidence_tmp[i] > min_confidence_tmp[i]:
                    confidence_tmp.append(max_confidence_tmp[i])
                    _dat_output.append((dat_predict[i, 0] - cp_tmp) / cp_tmp)
                else:
                    confidence_tmp.append(min_confidence_tmp[i])
                    _dat_output.append((dat_predict[i, 1] - cp_tmp) / cp_tmp)
            self.confidence[_s] = np.array(confidence_tmp)
            self.dat_output[_s] = np.array(_dat_output)
            # print(np.dstack((self.dat_output, self.confidence)))
            # plot baseline and predictions
            plt.plot(self.dat_output[_s], 'r')
            plt.show()
        return self.dat_output, self.confidence


class SAforTest(object):

    @classmethod
    def prepare_sa(cls, __s):
        jieba.initialize()
        print("START %s" % datetime.datetime.now())  # 开始时间
        tl_path = "/home/share/qt_repo/TLNEWS/"
        tl_subpath = ['2014/', '2015/', '2016/']
        tg = mul_config.PCON_DICT[__s.split('1')[0]]
        _sa_dir = './traindata/' + mul_config.PERIOD + '/' + mul_config.EXCHANGE + '/' + __s + '.sa.csv'
        with open(_sa_dir, 'w') as sa_file:
            writer = csv.writer(sa_file)
            header = ['datetime', 'sentiment score']
            writer.writerow(header)

        for tl_dir in tl_subpath:
            _path = tl_path + tl_dir
            tl_filelist = os.listdir(_path)
            for tl_file in tl_filelist:
                file_name = str(tl_file)
                print file_name
                url = _path + file_name
                f = open(url, "rb")
                content = f.read().decode('gbk', errors='ignore')
                pattern = re.compile('<start>.*?<lab(.*?)lab/><title(.*?)title/><time(.*?)time/><body(.*?)body/><end/>',
                                     re.S)
                reg_list = re.findall(pattern, content)

                for co in reg_list:
                    words = jieba.cut(co[3])
                    for word in words:
                        if word == tg.decode('utf-8'):
                            with open(_sa_dir, 'a+') as sa_file:
                                writer = csv.writer(sa_file)
                                row = [co[2].encode('utf-8'), co[0].encode('utf-8')]
                                writer.writerow(row)
                            break
        print("END %s" % datetime.datetime.now())


class SAtrain(object):

    @classmethod
    def train(cls):
        jieba.initialize()
        content = open("news714_2train.txt", "r").read()
        pattern = re.compile(r'<start>.*?<lab(.*?)lab/><body(.*?)body/><end/>', re.S)
        content_list = re.findall(pattern, content)
        idx = 0
        sentences = []
        train_arrays = []
        train_labs = []
        for __c in content_list[:]:
            words = " ".join(jieba.cut(__c[1]))
            tag_doc = gensim.models.doc2vec.TaggedDocument(list(words), [idx])
            sentences.append(tag_doc)
            idx += 1

        print("START AT %s" % datetime.datetime.now())
        model = gensim.models.Doc2Vec(sentences, size=100, window=5, min_alpha=0.025, alpha=0.025, min_count=2)
        for _ in xrange(10):
            model.train(sentences)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        print("END AT %s" % str(datetime.datetime.now()))
        model.save('model_doc2vec.txt')

        for num in xrange(len(content_list)):
            docvec = model.docvecs[num]  # paragraph vector
            train_arrays.append(docvec)
        for __c in content_list[:]:
            sa = string.atof(__c[0])
            train_labs.append(sa)

        model_lr = LinearRegression().fit(train_arrays, train_labs)
        with open('model_lr.pkl', 'wb') as lr_file:
            pickle.dump(model_lr, lr_file)


class SAinf(object):

    @classmethod
    def inf(cls, __s):
        model = gensim.models.Doc2Vec.load('model_doc2vec.txt')
        with open('model_lr.pkl', 'rb') as lr_file:
            lr_model = pickle.loads(lr_file.read())
        print("Model loaded.")
        jieba.initialize()
        test_arr = []
        time_arr = []
        conn = MySQLdb.connect(host="127.0.0.1", user="root", passwd="", charset='utf8')
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("use news_crawl")
        cursor.execute("select content,time from news_table")
        results = cursor.fetchall()
        for result in results:
            s = result['content']
            words = " ".join(jieba.cut(s))
            test_arr.append(model.infer_vector(words))
            time_arr.append(str(result['time']))

        sa_arr = lr_model.predict(test_arr)
        _sa_dir = './traindata/' + mul_config.PERIOD + '/' + mul_config.EXCHANGE + '/' + __s + '.sa.csv'
        with open(_sa_dir, 'w') as sa_file:
            writer = csv.writer(sa_file)
            header = ['datetime', 'sentiment score']
            writer.writerow(header)
            for i in xrange(len(sa_arr)):
                row = [time_arr[i], sa_arr[i]]
                writer.writerow(row)


class NewsCrawler(object):

    def __init__(self, aim='大豆', url='http://futures.eastmoney.com/', crawl_all=False):
        self.aim = aim
        self.url = url
        self.crawl_all = crawl_all

    @classmethod
    def crawl(cls, __url):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) '}
        try:
            request = urllib2.Request(__url, headers=headers)
            response = urllib2.urlopen(request)
            content = response.read()
            return content
        except urllib2.URLError, e:
            if hasattr(e, "code"):
                print(e.code)
            if hasattr(e, "reason"):
                print(e.reason)

    def run(self):
        content = self.crawl(self.url)
        # 获取目标期货的 url
        sql = r'<div.*?>期货</a>.*?全部品种</li>.*?<a href=.*?大商所</a>.*?<span.*?铁矿石</a><a.*?href="(.*?)".*?' \
              + str(self.aim) + r'</a>'
        # print content
        pattern = re.compile(sql, re.S)
        items = re.findall(pattern, content)
        url1 = items[0]
        # 获取当前日期
        nowtime = datetime.datetime.now().strftime('%Y%m%d')
        print "当前时间：", nowtime

        # 根据目标期货的url获取每条新闻的url及标题
        content1 = self.crawl(url1)
        sql1 = str(self.aim) + r'资讯</a>(.*?)' + str(self.aim) + r'评论</a>'
        pattern1 = re.compile(sql1, re.S)
        items1 = re.findall(pattern1, content1)
        news_type = 0
        txt = ''
        # print items1
        for item in items1:
            txt += item

        sql2 = r'<li><a\s+href="(.*?)".*?title="(.*?)".*?</a></li>'
        pattern2 = re.compile(sql2, re.S)
        items2 = re.findall(pattern2, txt)

        # 创建数据库
        new_db_name = "news_crawl"
        conn = MySQLdb.connect(host="localhost", user="root", passwd="", charset='utf8')
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("show databases")
        db_list = []
        for db in cursor.fetchall():
            db_list.append(db['Database'])
        if new_db_name not in db_list:
            cursor.execute("create database %s" % new_db_name)
        cursor.close()
        conn.close()
        # 创建数据表
        conn = MySQLdb.connect(host="127.0.0.1", user="root", passwd="", db="news_crawl", charset="utf8")
        cursor = conn.cursor()
        sql_ins = ('create table if not exists news_table('
                   'id int unsigned not null auto_increment primary key,'
                   'Topic char(20),'
                   'newsID char(20),'
                   'title char(100),'
                   'content TEXT(5000),'
                   'time char(30),'
                   'newstype int(5),'
                   'source char(20),'
                   'nowtime char(30))')
        cursor.execute(sql_ins)
        cursor.execute('truncate table news_table')

        news_counter = 0  # 计数新闻
        for item2 in items2:
            url2 = item2[0]
            content2 = self.crawl(url2)
            title = "".join(item2[1])

            # newsID=re.findall(r'http.*?news.*?,\d{8}(.*?).html',url2)
            # newsID="".join(newsID)

            # sql5=r'<!--published at (.*?)-(.*?)-(\d\d).*?by .*?>'
            sql5 = r'http.*?news.*?,(\d{8}).*?.html'
            pattern5 = re.compile(sql5, re.S)
            tim = re.findall(pattern5, url2)
            for ti in tim:
                newstime = "".join(ti)
            print newstime

            if (nowtime == newstime) or ((nowtime != newstime) and self.crawl_all):
                news_counter += 1
                sql6 = r'http.*?news.*?,\d{8}(.*?).html'
                pattern6 = re.compile(sql6, re.S)
                news_id = re.findall(pattern6, url2)
                news_id = "".join(news_id)

                # 获取新闻的时间和来源
                sql3 = r'<div class="time-source">.*?' \
                       r'<div class="time">(.*?)</div>' \
                       r'.*?source.*?来源.*?<img.*?alt="(.*?)".*?>'
                pattern3 = re.compile(sql3, re.S)
                items3 = re.findall(pattern3, content2)
                for item3 in items3:
                    # item3[0]时间
                    # item3[1]来源
                    time = "".join(str(item3[0]))
                    time = str(time)
                    time = datetime.datetime.strptime(time, '%Y年%m月%d日 %H:%M')
                    time = time.strftime('%Y-%m-%d %H:%M:%S')
                    print time
                    source = "".join(item3[1])

                # 新闻主体内容获取
                sql4 = '<p>　　(.*)</p>.*?<p class=.*?>'
                pattern4 = re.compile(sql4, re.S)
                content3 = re.findall(pattern4, content2)
                news_content = "".join(content3)
                # print type(news_content)
                print "News Title :: ", title

                insert_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                cursor.execute("insert into news_table "
                               "(Topic,content,time,newstype,source,newsID,title,nowtime) "
                               "values "
                               "('%s','%s','%s','%s','%s','%s','%s','%s')"
                               % (self.aim, news_content, time, news_type, source, news_id, title, insert_time))
        conn.commit()
        conn.close()


if __name__ == '__main__':
    ss = ["C1701", 'C1703']
    Train(ss[0]).run()
    #Predict(ss).run()
    # SAforTest.prepare_sa(ss[0])
