# -*- coding: utf-8 -*-
##
# @file manytomany.py
# @brief 多个数据, 多组合策略
# @author wondereamer
# @version 0.2
# @date 2015-12-09


#from quantdigger.engine.series import NumberSeries
#from quantdigger.indicators.common import MA
#from quantdigger.util import  pcontract
from quantdigger import *

class DemoStrategy(Strategy):
    """ 策略A1 """
    
    def on_init(self, ctx):
        """初始化数据""" 
        ctx.ma10 = MA(ctx.close, 10, 'ma10', 'y', 2) #, 'ma20', 'b', '1')
        ctx.ma20 = MA(ctx.close, 20, 'ma20', 'b', 2) #, 'ma20', 'b', '1')
        self.candicates = []

    def on_symbol(self, ctx):
        if ctx.curbar > 20:
            if ctx.ma10[1] < ctx.ma20[1] and ctx.ma10 > ctx.ma20:
                self.candicates.append([ctx.symbol, 'buy'])
            elif ctx.ma10[1] > ctx.ma20[1] and ctx.ma10 < ctx.ma20:
                self.candicates.append([ctx.symbol, 'sell'])

    def on_bar(self, ctx):
        if self.candicates:
            print self.candicates
            if self.candicates[0][1] == 'buy':
                ctx.buy(ctx.close, 1, self.candicates[0][0])
                print('策略%s, 买入%s' % (ctx.strategy, self.candicates[0][0]))
            elif self.candicates[0][1] == 'sell' and ctx.position() > 0:
                ctx.sell(ctx.close, 1, self.candicates[0][0])
                print('策略%s, 卖出%s' % (ctx.strategy, self.candicates[0][0]))

            self.candicates = []

    def on_exit(self, ctx):
        return

if __name__ == '__main__':
    from quantdigger.digger import finance

    set_symbols(['B.SHFE-1.DAY',
                 # 'A.SHFE-1.DAY',
                 'AG.SHFE-1.DAY'])
    profile = add_strategy([DemoStrategy('001')], {'capital': 10000000})
    #comb2 = add_strategy([DemoStrategy('B1'), DemoStrategy2('B2')], {'capital': 20000000, 'ratio': [0.4, 0.6]})
    run()
    # 打印组合1的统计信息
    curve1 = finance.create_equity_curve(profile.all_holdings())
    print '组合A', finance.summary_stats(curve1, 252*4*60)

    # 绘制k线，交易信号线
    from quantdigger.digger import finance, plotting

    s = 0
    # 绘制策略A1, 策略A2, 组合的资金曲线
    plotting.plot_strategy(profile.data(0), profile.technicals(0),
                            profile.deals(0), curve1.equity.values,
                            profile.marks(0))
