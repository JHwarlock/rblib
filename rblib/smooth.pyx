# -*- coding: UTF-8 -*-
import numpy as np
from scipy import interpolate

def sm_cv(data,winsize=10):
	"Convolution smoothing"
	window = np.ones(np.int(winsize)) / float(winsize)
	return np.convolve(data,window,'same')

def sm_spline(xdata,ydata,winsize=8):
	"Univariate Spline"
	sy = interpolate.UnivariateSpline(xdata, ydata, s=8)(xdata) # use xdata to get smooth y ,for univaite
	return sy
"""

参数插值：
前面所介绍的插值函数都需要X轴的数据是按照递增顺序排列的，就像一般的y=f(x)函数曲线一样。数学上还有一种参数曲线，它使用一个参数t和两个函数x=f(t), y=g(t)，定义二维平面上的一条曲线，例如圆形、心形等曲线都是参数曲线。参数曲线的插值可以通过splprep()和splev()实现，这组函数支持高维空间的曲线的插值，这里以二维曲线为例，介绍其用法。
1首先调用splprep()，其第一个参数为一组一维数组，每个数组是各点在对应轴上的坐标。s参数为平滑系数，与UnivariateSpline的含义相同。splprep()返回两个对象，其中tck是一个元组，其中包含了插值曲线的所有信息。t是自动计算出的参数曲线的参数数组。
2调用splev()进行插值运算，其第一个参数为一个新的参数数组，这里将t的取值范围等分200份，第二个参数为splprep()返回的第一个对象。实际上，参数数组t是正规化之后的各个线段长度的累计，因此t的范围位0到1。
其结果如图所示，图中比较了平滑系数为0和1e-4时的插值曲线。当平滑系数为0时，插值曲线通过所有的数据点。
x = [ 4.913,  4.913,  4.918,  4.938,  4.955,  4.949,  4.911, 4.848,  4.864,  4.893,  4.935,  4.981,  5.01 ,  5.021]

y = [ 5.2785,  5.2875,  5.291 ,  5.289 ,  5.28  ,  5.26  ,  5.245 , 5.245 ,  5.2615,  5.278 ,  5.2775,  5.261 ,  5.245 ,  5.241]
"""

def sm_spl(xdata,ydata,winsize=0.5):
	tck, t = interpolate.splprep([xdata, ydata], s=winsize)
	sx, sy = interpolate.splev(np.linspace(t[0], t[-1], 200), tck)
	return sx,sy


def sm_define(data,winsize=10,m="min"):
	lendata = len(data)
	winsize = min(winsize,lendata)
	newdata = np.zeros(lendata)
	if m == "median":
		fun = np.median
	elif m == "mean":
		fun = np.mean
	elif m == "min":
		fun = np.min
	else:
		fun = None
	for i in range(lendata-winsize):
		newdata[i] = fun(data[i:i+winsize])
	for i in range(lendata-1,lendata-1-winsize,-1):
		newdata[i] = fun(data[i+1-winsize:i+1])
	return newdata

