import numpy as np
from rblib import mplconfig,statplot
import matplotlib.pyplot as plt
import seaborn as sns

## sns.set_style("whitegrid")

def kdeplot(x1,x2,xlabel='x',ylabel='y',cbarlabel="Density",cmap="Reds", shade=True, shade_lowest=False,kernel='gau',cbar=True,fig_prefix="testkde",scatter=False,svg=1,xlim=None,ylim=None):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax = sns.kdeplot(x1, x2,cmap=cmap, shade=shade, shade_lowest=shade_lowest,cbar=cbar,cbar_kws={"label":cbarlabel})
	if scatter:
		ax.plot(x1,x2,'bo',markersize=3,linewidth=0,alpha=0.4)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if xlim is not None:
		ax.set_xlim(xlim[0],xlim[1])
	if ylim is not None:
		ax.set_ylim(ylim[0],ylim[1])
	plt.savefig(fig_prefix+".png",format='png',dpi=300)
	if svg:
		plt.savefig(fig_prefix+".svg",format='svg',dpi=300)
	plt.clf()
	plt.close()
	return 0

# [x1,x2],[x3,x4]

def kdeplot_mul(data,xlabel='x',ylabel='y',cbarlabel="Density",cmap=["Blues","Reds"],shade=True,shade_lowest=False,kernel='gau',cbar=True,fig_prefix="testkde",scatter=False,svg=1,xlim=None,ylim=None,n_levels=10,figsize=(6,6)):
	fig = plt.figure(figsize=(6,6))
	ax = fig.add_subplot(111)
	idx = 0 
	for x1,x2 in data:
		ax = sns.kdeplot(x1, x2,cmap=cmap[idx], shade=shade, shade_lowest=shade_lowest,cbar=cbar,cbar_kws={"label":cbarlabel},n_levels=n_levels)
		if scatter:
			ax.plot(x1,x2,'bo',markersize=3,linewidth=0,alpha=0.4)
		idx += 1
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if xlim is not None:
		ax.set_xlim(xlim[0],xlim[1])
	if ylim is not None:
		ax.set_ylim(ylim[0],ylim[1])
	plt.savefig(fig_prefix+".png",format='png',dpi=300)
	if svg: plt.savefig(fig_prefix+".svg",format='svg',dpi=300)
	plt.clf()
	plt.close()
	return 0

if __name__ == "__main__":
	x = np.random.multivariate_normal([1,1],[[1,0.8],[0.8,1]],size=1000)
	x1 = x[:,0]
	x2 = x[:,1]
	kdeplot(x1,x2,scatter=True)

