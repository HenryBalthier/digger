# -*- coding:utf-8 -*-
import urllib2
import re
import datetime
import MySQLdb


class Crawler(object):

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

        # 创建数据库及表
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


                # print '*'*100
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
    c = Crawler()
    c.run()

