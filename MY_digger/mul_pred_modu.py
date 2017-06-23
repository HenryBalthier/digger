# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
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
#import matplotlib.pyplot as plt

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


class Train(object):

    def __init__(self, _s):
        data_frame = []
        self.s = _s

        self.train_dir = './traindata/' + mul_config.PERIOD + '/' \
                         + mul_config.EXCHANGE + '/' + self.s + '.train.csv'
        if not os.path.exists(self.train_dir):
            SplitFile(self.s).split()
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
        df = DataMerger(data_frame[1:], sa_frame[1:]).merge()
        self.dataset = df
        self.dataset = self.dataset.astype('float32')
        self.close_idx = ML_Default.get_close_idx()   # index of the column of close price
        self.high_idx = ML_Default.get_high_idx()
        self.low_idx = ML_Default.get_low_idx()
        self.num_steps = 30
        self.pred_range = ML_Default.get_pred_range()  # if 1, predict the next period close price
        self.fluq_n = 0
        self.train_data = self.dataset[:, :]

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
        model.add(Dense(target_dim, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer="adam",
                      metrics=["mean_absolute_percentage_error"])
        model.fit(train_x, train_y, batch_size=batch_size, validation_split=0.2,
                  nb_epoch=num_epoch, verbose=2)
        pred_x = pred_x.reshape((1, self.num_steps, feature_dim))
        _fluq_n = model.predict_on_batch(pred_x)
        _fluq_n = scaler_fluq.inverse_transform(_fluq_n)
        return _fluq_n[0, 0]

    def _dat_for_price_pred(self, _dat, _num_steps=30, _pred_range=1):
        dat_x = []
        dat_y = []
        # normalize the dataset
        for _i in xrange(_dat.shape[1]):
            scaler_pr = StandardScaler().fit(_dat[:, _i].reshape(-1, 1))
            _dat[:, _i] = scaler_pr.transform(_dat[:, _i].reshape(-1, 1)).reshape(-1,)

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

    def run_fluq_n(self):
        self.run()
        return self._cal_fluq_n()

    def run(self):

        feature_dim = self.train_data.shape[1]
        train_x, train_y = self._dat_for_price_pred(self.train_data, self.num_steps, self.pred_range)
        target_dim = 1
        # reshape input and output to be [samples, time steps, features]
        train_x = train_x.reshape((train_x.shape[0], self.num_steps, feature_dim))
        train_y = train_y.reshape((train_y.shape[0], target_dim))
        print("Data loaded.")

        in_neurons = feature_dim
        out_neurons = target_dim
        hidden_neurons = 100
        batch_size = 100
        num_epoch = 100
        model = Sequential()
        # model.add(LSTM(hidden_neurons, input_dim=in_neurons, return_sequences=True))
        # model.add(TimeDistributed(Dense(out_neurons, activation="linear")))
        model.add(LSTM(hidden_neurons, input_dim=in_neurons, return_sequences=False))
        model.add(Dense(out_neurons, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer="adam",
                      metrics=["mean_absolute_percentage_error"])
        print("Model compiled.")
        model.fit(train_x, train_y, batch_size=batch_size, validation_split=0.2,
                  nb_epoch=num_epoch, verbose=2)
        print("Model trained.")
        filename = mul_config.PERIOD + '_' + self.s + '.h5'
        model.save(filename)
        print("Model saved.")


class Predict(object):
    # prepare data
    # this should be replaced by new price data of the same future.
    # expecting: dat_input = [open, close, high, low, volume]

    def __init__(self, _s_list):

        self.s_list = _s_list
        self.dat_output = {}
        self.confidence = {}

    def run(self):
        for _s in self.s_list:
            data_frame = []
            test_dir = './data/' + mul_config.PERIOD + '/' + mul_config.EXCHANGE + '/' + _s + '.csv'
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
            df = DataMerger(data_frame[1:], sa_frame[1:]).merge()
            dat_input = df
            dat_input = dat_input.astype('float32')
            confidence_lookback = 5
            close_idx = ML_Default.get_close_idx()
            pred_range = ML_Default.get_pred_range()
            scaler = []
            # normalize the dataset
            for i in xrange(5):
                scaler.append(StandardScaler().fit(dat_input[:, i].reshape(-1, 1)))
                dat_input[:, i] = scaler[i].transform(dat_input[:, i].reshape(-1, 1)).reshape(-1,)
            print("Data loaded.")

            # load model
            filename = mul_config.PERIOD + '_' + _s + '.h5'
            model = load_model(filename)
            print("Model loaded.")

            feature_dim = dat_input.shape[1]
            model_input = dat_input.reshape((dat_input.shape[0], 1, feature_dim))

            dat_predict = []
            dat_mape = []    # mean_absolute_percentage_error
            for i in xrange(confidence_lookback):
                dat_mape.append(100)    # for confidence generation
            for i in xrange(model_input.shape[0]):
                dat_x = model_input[i, :, :].reshape(1, 1, feature_dim)
                dat_predict.append(model.predict_on_batch(dat_x))
                if i <= model_input.shape[0] - pred_range:
                    tmp_i = i + 1
                    dat_max = np.max(model_input[tmp_i:(tmp_i + pred_range), :, close_idx], axis=0)
                    dat_min = np.min(model_input[tmp_i:(tmp_i + pred_range), :, close_idx], axis=0)
                    _tmp = model_input[i, :, close_idx]
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
            close_price = dat_input[:, close_idx]
            _dat_output = (dat_predict - close_price) / close_price
            self.dat_output[_s] = _dat_output
            dat_mape = 100 - np.array(dat_mape)
            confidence_tmp = []
            for i in xrange(dat_predict.shape[0]):
                confidence_tmp.append(np.mean(dat_mape[i: i + confidence_lookback]))
            self.confidence[_s] = np.array(confidence_tmp)
            #print(np.dstack((self.dat_output, self.confidence)))
            # plot baseline and predictions
            #plt.plot(close_price, 'b')
            #plt.plot(dat_predict, 'r')
            #plt.show()
        #return self.dat_output, self.confidence


class DataMerger(object):
    def __init__(self, f_data, sa_data):
        self.f_data = pd.DataFrame(f_data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        __sa_data = pd.DataFrame(sa_data, columns=['datetime', 'sa'])
        # print(__sa_data['sa'])
        __sa_data['sa'] = pd.to_numeric(__sa_data['sa'], errors='coerce')
        # print(__sa_data['sa'])
        __sa_grouped = __sa_data.groupby('datetime', as_index=False)
        self.sa_data = pd.DataFrame(__sa_grouped.mean())
        #print(self.sa_data)

    def merge(self):
        df = pd.merge(self.f_data, self.sa_data, how='left', on='datetime')
        df = df.fillna('0')
        return df.values[:, 1:]


class SAforTest(object):

    @classmethod
    def prepare_sa(cls, __s):
        jieba.initialize()
        print("START %s" % datetime.datetime.now())  # 开始时间
        tl_path = "/home/share/qt_repo/TLNEWS/"
        tl_subpath = ['2014/', '2015/', '2016/']
        tg = mul_config.PCON_DICT[__s[:-4]]
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


class SplitFile(object):
    def __init__(self, _s):
        self.s = _s
        self.fname = './data/' + mul_config.PERIOD + '/' \
                     + mul_config.EXCHANGE + '/' + self.s  + '.csv'   # file name
        self.split_ratio = ML_Default.get_split_ratio()   # ratio for training data

    def split(self):
        with open(self.fname) as f:
            for length, _ in enumerate(f):
                pass
        train_filesize = int(length * self.split_ratio)
        train_filename = './traindata/' + mul_config.PERIOD + '/' \
                         + mul_config.EXCHANGE + '/' + self.s + '.train.csv'   # file name
        with open(self.fname) as f:
            lines = f.readlines()
            with open(train_filename, 'w') as train_file:
                train_file.writelines(lines[:train_filesize])


if __name__ == '__main__':
    ss = ["AG", "AG"]
    do, co = Predict(ss).run()
    print(do)
    print(co)

