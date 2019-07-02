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

if __name__ == "__main__":
	data = pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/MASS/birthwt.csv")
	Y = data["low"]
	X = data[["age","lwt","smoke","ht","ui","ftv"]]
	rslt = logisticregress(Y,X)
	print rslt.llr_pvalue
	print rslt.params
	print rslt.pvalues
