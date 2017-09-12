
# digger
# 一个基于quantdigger0.3的期货回测模块

最新quantdigger地址：[基于python的量化交易平台 （作者：QuantFans）](https://github.com/QuantFans/quantdigger)

以下是基于quantdigger0.3版本搭建的一套期货策略+回测系统，加入了机器学习对行情的辅助分析，使用的传统的海龟逻辑交易策略。

# 相关依赖
* quantdigger0.3及以上
* anaconda
* karas
* mysql等

# 主要功能
* 通过quantdigger实现了期货交易回测的基本框架
* mytest中实现了[海龟逻辑交易策略](https://baike.baidu.com/item/%E6%B5%B7%E9%BE%9F%E4%BA%A4%E6%98%93%E6%B3%95%E5%88%99/10565764?fr=aladdin "")和结合机器学习辅助预测的策略等5种
* base中实现了风险控制、头寸计算，6种指标周期长度、以及定义了回测过程中的交易记录等功能
* config中定义了基本的参数指标，包括测试合约、数据周期、期货相关性约束矩阵等
* pred中对历史数据进行划分学习，并总结规律预测下一个时间单位的可能情况
