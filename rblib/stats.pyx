import sys
import numpy as np
import  pandas as pd 
from statsmodels.tools import tools
from statsmodels.discrete.discrete_model import Logit

def logisticregress(Y,X,intercept=True):
	X = X.loc[:,X.std() != 0]
	#print X.std()
	#print X.head()
	if X.empty: return None
	if intercept:
		X = tools.add_constant(X)
	try:
	#print Y
	#print X
		md = Logit(Y,X,missing="drop") # Available options are none, drop, and raise. If none, no nan checking is done. If drop, any observations with nans are dropped. If raise, an error is raised. Default is none.
		rslt = md.fit(method="lbfgs",maxiter=1000,full_output=True,disp=True,retall=True)
		#print rslt.summary()
		sys.stderr.write(rslt.summary().as_text()+"\n")
	except Exception,e:
		sys.stderr.write(str(e)+"\n")
		return None
	return rslt # rslt.pvalues for each vars, llr_pvalue : float The chi-squared probability of getting a log-likelihood ratio statistic greater than llr.  llr has a chi-squared distribution with degrees of freedom `df_model`.

def get_xrange(data,extra=0.2,num = 200):
	"""
	Compute the x_range, i.e., the values for which the 
	density will be computed. It should be slightly larger than
	the max and min so that the plot actually reaches 0, and
	also has a bit of a tail on both sides.
	"""
	nmin = np.nanmin(data)
	nmax = np.nanmax(data)
	try:
		sample_range = nmax - nmin
	except ValueError:
		return []
	if sample_range < 1e-6:
		return np.asarray([nmin, nmax])
	return np.linspace(nmin - extra*sample_range, nmax + extra*sample_range, num)

def get_middle(x):
	return (x[0:-1] + x[1:])/2

import statsmodels.api as sm
def kde_density(y,x_range,kernel='gau',bw="normal_reference",seed=None):
	if seed is not None: np.random.seed(seed)
	dens = sm.nonparametric.KDEUnivariate(y)
	'''
	“biw” for biweight
	“cos” for cosine
	“epa” for Epanechnikov
	“gau” for Gaussian.
	“tri” for triangular
	“triw” for triweight
	“uni” for uniform

	“scott” - 1.059 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
	“silverman” - .9 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
	“normal_reference” - C * A * nobs ** (-1/5.), where C is calculated from the kernel. Equivalent (up to 2 dp) to the “scott” bandwidth for gaussian kernels. See bandwidths.py
	If a float is given, it is the bandwidth.
	fft is True  for gau
	'''
	fft = True if kernel == "gau" else False
	dens.fit(kernel=kernel,bw=bw,fft=fft)
	y = np.zeros_like(x_range)
	for i in range(len(x_range)):
		y[i] = dens.evaluate(x_range[i])[0]
	return y

def get_density(values,method=["counts",],bins=20,extra=0.2):
	assert method in ["counts","normalized_counts","kde",]
	x_range = get_xrange(values,extra=extra)
	if method == "counts":
		y, bin_edges = np.histogram(values, bins=bins, range=(min(x_range), max(x_range)))
		x_range = get_middle(bin_edges)
	elif method == "normalized_counts":
		y, bin_edges = np.histogram(values, bins=bins, range=(min(x_range), max(x_range)),density=False)
		y = y/len(values)
		x_range = get_middle(bin_edges)
	elif method == "kde":
		y = kde_density(values,x_range)
	return x_range,y

if __name__ == "__main__":
	data = pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/MASS/birthwt.csv")
	Y = data["low"]
	X = data[["age","lwt","smoke","ht","ui","ftv"]]
	rslt = logisticregress(Y,X)
	print(rslt.llr_pvalue)
	print(rslt.params)
	print(rslt.pvalues)
