# -*- coding: UTF-8 -*-
import sys
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from  rklib.utils import dirDetectCreate 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import font_manager as fm
from matplotlib_venn import venn2,venn3
import itertools
import mutilstats
import scipy.cluster.hierarchy as sch


# for projection='3d'
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import fcluster; import pandas
###动态设置字体

###
from matplotlib.patches import Polygon
# to get kmeans and scipy.cluster.hierarchy
from scipy.cluster.vq import *
from scipy.cluster.hierarchy import *

###
from matplotlib.colors import LogNorm

##kmeans归一化处理 from scipy.cluster.vq import whiten
from scipy.cluster.vq import whiten
#mpl.style.use('ggplot')

import mplconfig
from mplconfig import styles,color_grad,rgb2hex,inscolor
from statplot import clean_axis
import pandas as pd
import traits

mpl.rcParams['axes.facecolor'] = u"#FFFFFF"
### to plot mutation profile in 
mpl.rcParams['ytick.direction'] = "out"

def retain_x(ax):
	ax.get_yaxis().set_ticks([])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')

def retain_y(ax):
	"""
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	# Only show ticks on the left and bottom spines
	ax1.yaxis.set_ticks_position('left')
	ax1.xaxis.set_ticks_position('bottom')
	"""
	ax.get_xaxis().set_ticks([])
	#ax.get_yaxis().set_ticks([])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	#for spx in ax.spines.values():
	#	spx.set_visible(False)
def framey(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.get_xaxis().set_ticks([])
	#ax.get_yaxis().set_ticks([])	
	for l in ax.get_xticklines() + ax.get_yticklines():
		l.set_markersize(0)
def mutation_spec(snnum,targetgenelist,dfgroup,groups,mutlist,genemutdf,fig_prefix="testplot",width=0.84,CNV=None,cnvlist=None,unkcolor=u"#E0E0E0",cmtmp="Dark2"): # input shold first to sort
	fig = plt.figure(figsize=(8,6))
	#mapGS = gridspec.GridSpec(12,1,wspace=0.0,hspace=0.0,height_ratios=[2,1,0.1,1,0.1,1,0.1,1,0.1,1,0.1,1]) ### 
	num_traits_c = len(groups)
	plotarr = [17,0.2] # 1 use for legend for mutation , 
	

	plotarrays = [1,0.1]*(num_traits_c-1)+[1,0.2]+plotarr + (num_traits_c + 2) * [1,]
	mutpos = num_traits_c * 2 
	GS = gridspec.GridSpec(len(plotarrays),1,wspace=0.0,hspace=0,height_ratios=plotarrays)
	xlocations = np.arange(snnum)
	ones = np.ones(snnum) 
	######################## plot the group bars 
	idx = 0
	tmpidxcolor = 0
	
	allgroups = []
	for i in xrange(len(groups)):
		group = groups[i]
		setsortgroups = sorted(set(dfgroup[group]))
		allgroups = allgroups + setsortgroups
	allgroups = list(set(allgroups))
	hcolors = inscolor(allgroups,cmtmp)
	hcolors["Unk"] = unkcolor
	hcolors["unk"] = unkcolor
	for i in xrange(len(groups)):
		group = groups[i]
		ax = fig.add_subplot(GS[idx*2])
		#ax.broken_barh([(110, 10),(120,10),(130,10),(150, 10)], (10, 9), facecolors='blue',alpha=0.5)
		#ret_xlocations_x,setsortgroups = groupssplit(xlocations_x,hc[key])
		setsortgroups = sorted(set(dfgroup[group])) #### use "unk" as gray ## 9B9B9B
		#colors = styles(len(setsortgroups)+tmpidxcolor,cm.Accent)[0]
		#colors = colors[tmpidxcolor:]
		#if "Unk" in setsortgroups:
		#	colors[setsortgroups.index("Unk")] = unkcolor
		tmpidxcolor += len(setsortgroups)
		legends = []
		for k in xrange(len(setsortgroups)):
			#l = ax.broken_barh(ret_xlocations_x[i],(0,1),facecolors=colors[i],alpha=0.5)
			groupname = setsortgroups[k]
			tmpidx = dfgroup[group]==groupname
			if np.sum(tmpidx.values) <= 0:continue
			l = ax.bar(xlocations[tmpidx.values],ones[tmpidx.values],width,color=hcolors[setsortgroups[k]],linewidth=0,alpha=1.0,align='center')
			legends.append(l)
		ax.set_ylabel(group,rotation=0,horizontalalignment="right",verticalalignment='center',fontsize=12)
		clean_axis(ax)
		ax.set_xlim(xlocations[0]-width/2.0,xlocations[-1]+width/2.0)
		ax = fig.add_subplot(GS[mutpos+2+2 + idx]) # mutpos+3
		ax.legend(legends,setsortgroups,loc="center left",ncol=len(setsortgroups),fancybox=False,frameon=False,handlelength=0.75)
		clean_axis(ax)
		idx += 1
	##### plot mutation 
	#ngene,nsamples = genesframe.shape
	
	mutcolors = styles(len(mutlist),colorgrad="custom2")[0]
	mutcolors.append(unkcolor)
	cnvcolors = ["#0076AF",unkcolor,"#E31915"]
	stackvAX = fig.add_subplot(GS[mutpos])
	ngenesmut, nsamples= genemutdf.shape
	ngenes = len(targetgenelist) 
	ngenesCNV, nsamples= CNV.shape


	legends = []
	legendsnames = []
	legendscnv = []
	legendsnamescnv = []
	samplenames = genemutdf.columns
	genenames   = genemutdf.index
	htmp = {}
	cumtmp = np.zeros(len(xlocations))
	cnvnames = ["None","Amplification","Deletion"]
	for i in xrange(ngenes-1,-1,-1):
		#if i <> ngenes-1:
		if 1:
			genename = targetgenelist[i]
			if genename not  in genemutdf.index:continue
			for j in xrange(len(mutlist)):
				tmpidx = genemutdf.loc[genename] == mutlist[j]
				if tmpidx.sum() >= 1:
					rects = stackvAX.bar(xlocations[tmpidx.values],ones[tmpidx.values]*0.4,width,color=mutcolors[j],linewidth=0,alpha=1.0,bottom=cumtmp[tmpidx.values]+0.3,align='edge')
					if mutlist[j] not in htmp and mutlist[j] <> "A":
						legends.append(rects);legendsnames.append(mutlist[j]);htmp[mutlist[j]] = None
			if genename not in CNV.index:continue
			for j in xrange(len(cnvlist)):
				tmpidx =CNV.loc[genename] == cnvlist[j]
				if tmpidx.sum() >= 1:
					rects = stackvAX.bar(xlocations[tmpidx.values],ones[tmpidx.values]*0.9,width,color=cnvcolors[j],linewidth=0,alpha=0.5,bottom=cumtmp[tmpidx.values],align='edge')
					if cnvlist[j] not in htmp and cnvlist[j] <> 0:
						legendscnv.append(rects);legendsnamescnv.append(cnvnames[cnvlist[j]]);htmp[cnvlist[j]] = None
			cumtmp = cumtmp + np.ones(len(samplenames))
		"""
		else:
			cumtmp = np.zeros(len(xlocations))
			genename = targetgenelist[i]
			if genename not  in genemutdf.index:continue
			for j in xrange(len(mutlist)):
				tmpidx = genemutdf.loc[genename] == mutlist[j]
				if tmpidx.sum() >= 1:
					rects = stackvAX.bar(xlocations[tmpidx.values],ones[tmpidx.values]*0.4,width,color=mutcolors[j],linewidth=0,alpha=1.0,align='edge')
					if mutlist[j] not in htmp and mutlist[j] <> "A":
						legends.append(rects);legendsnames.append(mutlist[j]);htmp[mutlist[j]] = None
			if genename not in CNV.index:continue
			for j in xrange(len(cnvlist)):
				tmpidx = genemutdf.loc[genename] == cnvlist[j]
				if tmpidx.sum() >= 1:
					rects = stackvAX.bar(xlocations[tmpidx.values],ones[tmpidx.values]*0.9,width,color=cnvcolors[j],linewidth=0,alpha=1.0,align='edge') 
					if cnvlist[j] not in htmp and cnvlist[j] <> 0:
						legends.append(rects);legendsnames.append(cnvlist[j]);htmp[cnvlist[j]] = None
		"""	
	stackvAX.set_xlim(xlocations[0],xlocations[-1]+width)
	stackvAX.set_ylim(ngenes - 17,ngenes-0.1)
	stackvAX.set_yticks(np.arange(ngenes)+ 0.5)
	stackvAX.set_yticklabels(genenames[::-1])
	framey(stackvAX)
	ax = fig.add_subplot(GS[mutpos+2])
	ax.legend(legends,legendsnames,loc="center left", ncol=7,fancybox=False,frameon=False,numpoints=1,handlelength=0.75)
	clean_axis(ax)
	ax = fig.add_subplot(GS[mutpos+3])
	ax.legend(legendscnv,legendsnamescnv,loc="center left", ncol=7,fancybox=False,frameon=False,numpoints=1,handlelength=0.75)
	clean_axis(ax)
	
	#retain_x(stackvAX)
	fig.tight_layout()
	plt.savefig(fig_prefix+".png",format='png',dpi=600)
	plt.savefig(fig_prefix+".svg",format='svg',dpi=600)
	plt.clf()
	plt.close()
	return 0

if __name__ == "__main__":
	fsurf = sys.argv[1]
	stackdata = np.random.random((2,82))
	stacknames = ["Nonsynonymous","synonymous"]
	
	traitdata = traits.TraitInfo()
	traitdata.parse(sys.argv[1])
	datamut = []
	for i in xrange(10):
		datamut.append(["A",]*82)
	genemutdf = pd.DataFrame(np.asarray(datamut),index=["1","2","3","4","5","6","7","8","9","10"],columns=range(1,83))
	genemutdf.values[0,12] = "frame-shift"
	genemutdf.values[1,3]  = "in-frame-shift"
	genemutdf.values[4,20] = "splice"
	mutlist = ["frame-shift","in-frame-shift","splice","A"]	
	mutation_spec(82,traitdata.dframe,["Types","HPV","Stage","Samplegroup","Metastasis","Recurrence","Death"],stackdata,stacknames,traitdata,mutlist,genemutdf)

