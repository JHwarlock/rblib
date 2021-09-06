# -*- coding: UTF-8 -*-
import numpy as np
from scipy import interpolate

def smooth(x,winsize=11,window='hanning',correct=True):
	"""
	smooth the data using a window with requested size.
	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.
	
	input:
		x: the input signal 
		winsize: the dimension of the smoothing window; should be an odd integer
		window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
					flat window will produce a moving average smoothing.

	output:
		the smoothed signal
																				        
	example:
		t=linspace(-2,2,20)
		x=sin(t)+randn(len(t))*0.1
		y=smooth(x)
																											    
	see also: 
		numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
		scipy.signal.lfilter
																																 
	TODO: the window parameter could be the window itself if an array instead of a string
	NOTE: length(output) != length(input), to correct this: return y[(winsize/2):(winsize/2)+len(x)] instead of just y.
	"""
	if x.ndim != 1:
		raise ValueError, "smooth only accepts 1 dimension arrays."
	if x.size < winsize:
		raise ValueError, "Input vector needs to be bigger than window size."
	if winsize < 3:
		return x
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
	
	"""
	if x = [1,2,3,4,5,6,7,8,9,10,11] ; winsize = 5
	s = [5,4,3,2,1,2,3,4,5,6,7,8,9,10,11,10,9,8,7]
	"""
	s=np.r_[x[winsize-1:0:-1],x,x[-2:-winsize-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=np.ones(winsize,'d')
	else:
		w=eval('np.'+window+'(winsize)') # different weights method, https://scipy-cookbook.readthedocs.io/_static/items/attachments/SignalSmooth/smoothsignal.jpg 
	y=np.convolve(w/w.sum(),s,mode='valid')
	if correct:
		return y[int(winsize/2):int(winsize/2) + len(x)]
	else:
		return y

def sm_cv(data,winsize=11):
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


def sm_define(data,winsize=11,m="mean",f='line',window=None):
	lendata = len(data)
	winsize = min(winsize,lendata)
	assert m in ["median","mean","min","max"]
	fun = eval('np.'+ m) 
	# 此处，我们提供一种和smooth 不同的填充方式
	# if x = [1,2,3,4,5,6,7,8,9,10,11] ; winsize = 5
	# => 'line':  s = [1,1,1,1,1,2,3,4,5,6,7,8,9,10,11,11,11,11,11]
	# => 'cycle': s = [5,4,3,2,1,2,3,4,5,6,7,8,9,10,11,10,9,8,7]
	x = data
	if f == 'line':
		s = np.r_[[x[0],]*(winsize-1),x,[x[-1],]*(winsize-1)]
	elif f == 'cycle':
		s = np.r_[x[winsize-1:0:-1],x,x[-2:-winsize-1:-1]]
	newdata = []
	if window is None:
		for i in range(int(winsize/2),int(winsize/2) + lendata):
			newdata.append(fun(s[i:i+winsize]))
	else:
		assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
		w = eval('np.'+window+'(winsize)')	
		for i in range(int(winsize/2),int(winsize/2) + lendata):
			newdata.append(np.sum(w/w.sum() * s[i:i+winsize]))
	return newdata

