# -*- coding: utf-8 -*-
import sys
import gensim
import jieba
import pickle
import MySQLdb
import csv
from config import Default

sadir = Default.get_sadir()
reload(sys)
sys.setdefaultencoding('utf-8')


class SAinf(object):

    @classmethod
    def inf(cls):
        model = gensim.models.Doc2Vec.load('./model_doc2vec.txt')
        with open('model_lr.pkl', 'rb') as lr_file:
            lr_model = pickle.loads(lr_file.read())
        print("Model loaded.")
        jieba.initialize()
        test_arr = []
        time_arr = []
        conn = MySQLdb.connect(host="localhost", user="root", passwd="", charset='utf8')
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
        with open(sadir, 'w') as sa_file:
            writer = csv.writer(sa_file)
            header = ['datetime', 'sentiment score']
            writer.writerow(header)
            for i in xrange(len(sa_arr)):
                row = [time_arr[i], sa_arr[i]]
                writer.writerow(row)

if __name__ == "__main__":
    SAinf.inf()

