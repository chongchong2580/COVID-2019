# 拟合2019-nCov肺炎感染确诊人数

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def logistic_increase_function(t, K, P0, r):
    t0 = 11
    # t:time   t0:initial time    P0:initial_value    K:capacity  r:increase_rate
    exp_value = np.exp(r * (t - t0))
    return (K * exp_value * P0) / (K + (exp_value - 1) * P0)


fast_r = 0.40
slow_r = 0.64


def faster_logistic_increase_function(t, K, P0, ):
    return logistic_increase_function(t, K, P0, r=fast_r)


def slower_logistic_increase_function(t, K, P0, ):
    return logistic_increase_function(t, K, P0, r=slow_r)

#  日期及感染人数
# t=[11,18,19,20 ,21, 22, 23, 24,  25,  26,  27,  28,  29  ,30]
t = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
t = np.array(t)
# P=[41,45,62,291,440,571,830,1287,1975,2744,4515,5974,7711,9692]
P =  [1226, 2388, 2157, 1656, 1947, 2281, 2591, 2010, 1301, 1280,1217,1219,1228,1565,1803]
P = np.array(P)

# 用最小二乘法估计拟合
# popt, pcov = curve_fit(logistic_increase_function, t, P)
popt_fast, pcov_fast = curve_fit(faster_logistic_increase_function, t, P)
popt_slow, pcov_slow = curve_fit(slower_logistic_increase_function, t, P)
# 获取popt里面是拟合系数
print("K:capacity  P0:initial_value   r:increase_rate   t:time")
# print(popt)
# 拟合后预测的P值
# P_predict = logistic_increase_function(t,popt[0],popt[1],popt[2])
P_predict_fast = faster_logistic_increase_function(t, popt_fast[0], popt_fast[1])
P_predict_slow = slower_logistic_increase_function(t, popt_slow[0], popt_slow[1])
# 未来长期预测
# future=[11,18,19,20 ,21, 22, 23, 24,  25,  26,  27,28,29,30,31,41,51,61,71,81,91,101]
# future=np.array(future)
# future_predict=logistic_increase_function(future,popt[0],popt[1],popt[2])
# 近期情况预测
# tomorrow = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
tomorrow = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
tomorrow = np.array(tomorrow)
# tomorrow_predict=logistic_increase_function(tomorrow,popt[0],popt[1],popt[2])
tomorrow_predict_fast = logistic_increase_function(tomorrow, popt_fast[0], popt_fast[1], r=fast_r)
tomorrow_predict_slow = logistic_increase_function(tomorrow, popt_slow[0], popt_slow[1], r=slow_r)

# 绘图
plot1 = plt.plot(t, P, 's', label="confimed infected people number")
# plot2 = plt.plot(t, P_predict, 'r',label='predict infected people number')
# plot3 = plt.plot(tomorrow, tomorrow_predict, 's',label='predict infected people number')
plot2 = plt.plot(tomorrow, tomorrow_predict_fast, 's', label='predict infected people number fast')
plot3 = plt.plot(tomorrow, tomorrow_predict_fast, 'r')
plot4 = plt.plot(tomorrow, tomorrow_predict_slow, 's', label='predict infected people number slow')
plot5 = plt.plot(tomorrow, tomorrow_predict_slow, 'g')
plot6 = plt.plot(t, P_predict_fast, 'b', label='confirmed infected people number')

plt.xlabel('time')
plt.ylabel('confimed infected people number')

plt.legend(loc=0)  # 指定legend的位置右下角

print("32\n")
print(faster_logistic_increase_function(np.array(32), popt_fast[0], popt_fast[1]))
print(slower_logistic_increase_function(np.array(32), popt_slow[0], popt_slow[1]))

print("33\n")
print(faster_logistic_increase_function(np.array(33), popt_fast[0], popt_fast[1]))
print(slower_logistic_increase_function(np.array(33), popt_slow[0], popt_slow[1]))

plt.show()

print("Program done!")

#
# import numpy as np                                              # 导入数学函数
# from matplotlib import pyplot as plt                        # 导入作图函数
# from scipy.optimize import curve_fit as curve_fit      # 导入拟合函数
#
# # 数据录入——请在这里修改或补充每日病例数，数据太多时用"\"表示换行
# 每日病例数 = [101, 108, 41, 132, 116, 95, 92, 97, 110, 166, 124, 143, 104, 65, 163, 127, 55, 43, 23, 23, 19,18, 18, 24, 25, 39, 37, 54, 40, 27,\
#             36, 21, 12, 9, 13, 45, 65, 73, 7, 56, 40, 28, 26, 40, 46, 35, 40, 80, 101, 71, 59, 90, 85, 82, 93, 112, 87, 75,\
#             71, 54, 61, 102, 175, 214, 175, 233, 402, 397, 476, 1807, 1337, 3507, 1860, 1226, \
#             2388, 2157, 1656, 1947, 2281, 2591, 2010, 1301, 1280,1217,1219,1228,1565,1803,1787,2086]
#
# 天数 = len(每日病例数)                                  # 自动计算上面输入的数据所对应的天数
# xdata = [ i+1 for i in range(天数) ]                  # 横坐标数据，以第几天表示
# ydata = 每日病例数                                      # 纵坐标数据，表示每天对应的病例数
# plt.scatter(xdata, ydata, label='data')              # 把输入的数据用散点图列印出来
#
# # S型曲线函数公式定义
# def func(x, k, a, b):
#  return k/(1+(k/b-1)*np.exp(-a*x))
#
# # 非线性最小二乘法拟合
# popt, pcov = curve_fit(func, xdata, ydata, method='dogbox', \
#                         bounds=([1000., 0.01, 10.],[10000000., 1.0, 1000.]))
# k = popt[0]
# a = popt[1]
# b = popt[2]
#
# # 计算拟合数据后的数据
# 延长天数 = 40 # 需要预测的天数
# x = np.linspace(0, len(xdata)+延长天数)            # 横坐标取值
# y = func(x, *popt)                                          # 纵坐标计算值
#
# # 作图
# plt.plot(x, y,  color='r', label='fit')                      # 对拟合函数作图
# plt.xlabel('Day')                                              # 打印横坐标标签
# plt.ylabel('Number of Cases')                           # 打印纵坐标标签
# plt.title('A Rough Simulation and Prediction')     # 打印图表名称
# plt.legend(loc='best')                                      # 打印图例说明
# plt.show( )                                                    # 正式输出图表