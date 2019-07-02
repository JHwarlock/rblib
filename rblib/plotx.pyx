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
import pandas as pd
import traits

mpl.rcParams['axes.facecolor'] = u"#FFFFFF"
### to plot mutation profile in 
mpl.rcParams['ytick.direction'] = "out"

def clean_axis(ax):
	"""Remove ticks, tick labels, and frame from axis"""
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	for spx in ax.spines.values():
		spx.set_visible(False)
def retain_x(ax):
	ax.get_yaxis().set_ticks([])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')

def retain_xy(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	
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
	ax.get_xaxis().set_ticks([])
	#ax.get_yaxis().set_ticks([])	
	for l in ax.get_xticklines() + ax.get_yticklines():
		l.set_markersize(0)
def mutation_spec(snnum,dfgroup,groups,stackdata,stacknames,traitsurv,mutlist,genemutdf,fig_prefix="testplot",width=0.84,CNV=None,APOBEC=None,gainloss=None,cmtmp="Set2",figsize=(10,8)): # input shold first to sort
	fig = plt.figure(figsize=figsize)
	#mapGS = gridspec.GridSpec(12,1,wspace=0.0,hspace=0.0,height_ratios=[2,1,0.1,1,0.1,1,0.1,1,0.1,1,0.1,1]) ### 
	num_traits_c = len(groups)
	if isinstance(CNV,pd.DataFrame):
		plotarr = [3.5,0.08,1,3.5,0.08,1]
		mutpos = -6
		cnvpos = -3
	else:
		plotarr = [3.5,0.08,1] # 1 use for legend for mutation 
		mutpos = -3
	plotarrays = [1.2,0.15]+[1.2,0.1]+[0.25,0.08]*(num_traits_c-1)+[0.25,0.1]+plotarr
	GS = gridspec.GridSpec(len(plotarrays),2,wspace=0.0,hspace=0,height_ratios=plotarrays,width_ratios=[3.,1.2])
	#print plotarr[cnvpos]
	#xlocations,xlocations_x, width = getxlocations(snnum)
	xlocations = np.arange(snnum)
	ones = np.ones(snnum) 
	######################## plot the group bars 
	idx = 0
	
	tmpidxcolor = 0
	# process colors 
	allgroups = []
	for i in xrange(len(groups)):
		group = groups[i]
		setsortgroups = sorted(set(dfgroup[group]))
		allgroups = allgroups + setsortgroups
	allgroups = list(set(allgroups))
	hcolors = inscolor(allgroups,cmtmp)
	hcolors["Unk"] = u"#C1C1C1"
	hcolors["unk"] = u"#C1C1C1"
	for i in xrange(len(groups)):
		group = groups[i]
		ax = fig.add_subplot(GS[idx*2+4,0])
		#ax.broken_barh([(110, 10),(120,10),(130,10),(150, 10)], (10, 9), facecolors='blue',alpha=0.5)
		#ret_xlocations_x,setsortgroups = groupssplit(xlocations_x,hc[key])
		setsortgroups = sorted(set(dfgroup[group])) #### use "unk" as gray ## 9B9B9B
		#colors = styles(len(setsortgroups)+tmpidxcolor)[0]
		#colors = colors[tmpidxcolor:]
		#if "Unk" in setsortgroups:
		#	colors[setsortgroups.index("Unk")] = u"#C1C1C1"
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
		ax = fig.add_subplot(GS[idx*2+4,1])
		ax.legend(legends,setsortgroups,loc="center left",ncol=len(setsortgroups),fancybox=False,frameon=False,handlelength=0.75)
		clean_axis(ax)
		idx += 1
	######################## plot stack bars for  nonsysn and sysn
	n,p = stackdata.shape
	stackvAX = fig.add_subplot(GS[2,0])
	#colors = styles(len(stacknames))[0]
	colors = ["#0E5088","#43AF57"]
	legends = []
	for i in xrange(n):
		if i:
			cumtmp = cumtmp + np.asarray(stackdata[i-1,:])
			rects = stackvAX.bar(xlocations,stackdata[i,:],width,color=colors[i],linewidth=0.0,alpha=1.0,bottom=cumtmp,align='edge',label=stacknames[i])
			legends.append(rects)
		else:
			cumtmp = 0
			rects = stackvAX.bar(xlocations,stackdata[i,:],width,color=colors[i],linewidth=0.0,alpha=1.0,align='edge',label=stacknames[i])
			legends.append(rects)
	stackvAX.set_xlim(xlocations[0],xlocations[-1]+width)
	stackvAX.set_ylim(0,15)
	stackvAX.set_ylabel("Mutations/Mb",rotation=0,horizontalalignment="right",verticalalignment='center',fontsize=12)
	retain_y(stackvAX)
	ax = fig.add_subplot(GS[2,1])
	ax.legend(legends,stacknames,loc='center left', ncol=1,fancybox=False,frameon=False,handlelength=0.75)
	clean_axis(ax)

	######################### plot survival time for in barplot
	n,p = traitsurv.dframe.shape
	barax = fig.add_subplot(GS[0,0])
	colors = styles(2)[0]
	legends = []
	idx = traitsurv.dframe["Death"]==0	
	rects = barax.bar(xlocations[idx.values],traitsurv.dframe[idx]["Progression_free_survival"].values,width,color=colors[0],linewidth=0.0,alpha=1.0,align='edge')
	idx = traitsurv.dframe["Death"]==1
	rects = barax.bar(xlocations[idx.values],traitsurv.dframe[idx]["Progression_free_survival"].values,width,color=colors[1],linewidth=0.0,alpha=1.0,align='edge',label="Deceased")
	legends.append(rects)
	barax.set_xlim(xlocations[0],xlocations[-1]+width)
	barax.set_ylabel("PFS(Months)",rotation=0,horizontalalignment="right",verticalalignment='center',fontsize=12)
	retain_y(barax)
	ax = fig.add_subplot(GS[0,1])
	ax.legend(legends,["Deceased",],loc='center left', ncol=1,fancybox=False,frameon=False,numpoints=1,handlelength=0.75)
	clean_axis(ax)
	##### plot mutation 
	#ngene,nsamples = genesframe.shape
	mutcolors = styles(len(mutlist),colorgrad="custom2")[0]
	print mutcolors
	print mutlist
	#mutcolors.append("#C0C0C0")
	stackvAX = fig.add_subplot(GS[mutpos,0])
	ngenes, nsamples= genemutdf.shape
	legends = []
	legendsnames = []
	samplenames = genemutdf.columns
	genenames   = genemutdf.index
	htmp = {}
	for i in xrange(ngenes-1,-1,-1):
		if i <> ngenes-1:
			cumtmp = cumtmp + np.ones(len(samplenames))
			for j in xrange(len(mutlist)):
				tmpidx = genemutdf.loc[genenames[i]] == mutlist[j]
				if tmpidx.sum() >= 1:
					rects = stackvAX.bar(xlocations[tmpidx.values],ones[tmpidx.values]*0.9,width,color=mutcolors[j],linewidth=0,alpha=1.0,bottom=cumtmp[tmpidx.values],align='edge')
					if mutlist[j] not in htmp and mutlist[j] <> "A":
						legends.append(rects);legendsnames.append(mutlist[j]);htmp[mutlist[j]] = None
		else:
			cumtmp = 0
			for j in xrange(len(mutlist)):
				tmpidx = genemutdf.loc[genenames[i]] == mutlist[j]
				if tmpidx.sum() >= 1:
					rects = stackvAX.bar(xlocations[tmpidx.values],ones[tmpidx.values]*0.9,width,color=mutcolors[j],linewidth=0,alpha=1.0,align='edge')
					if mutlist[j] not in htmp and mutlist[j] <> "A":
						legends.append(rects);legendsnames.append(mutlist[j]);htmp[mutlist[j]] = None

	stackvAX.set_xlim(xlocations[0],xlocations[-1]+width)
	stackvAX.set_ylim(0,ngenes-0.1)
	stackvAX.set_yticks(np.arange(ngenes)+ 0.5)
	stackvAX.set_yticklabels(genenames[::-1])
	framey(stackvAX)
	ax = fig.add_subplot(GS[mutpos+2,0])
	ax.legend(legends,legendsnames,loc=0, ncol=7,fancybox=False,frameon=False,numpoints=1,handlelength=0.75)
	clean_axis(ax)
	
	### plot APOBEC attack, statckplot
	#stackvdata = APOBEC.values
	ylocations = np.arange(ngenes)
	#n,p = stackvdata.shape
	stackvAX = fig.add_subplot(GS[mutpos,1])
	linewidth = 0
	alpha=0.8
	width=0.8
	legends = ["Relapse","metastasis","no"] # chage to P
	#legends = ["Relapse","metastasis"]
	stackvdata = APOBEC.loc[legends].values
	print stackvdata
	n,p = stackvdata.shape
	for i in xrange(n):
		if i:
			print i
			cumtmp = cumtmp + np.asarray(stackvdata[i-1,:])
			rects = stackvAX.barh(ylocations,np.asarray(stackvdata[i,:]),width,color=hcolors[legends[i]],linewidth=linewidth,alpha=alpha,left=cumtmp,align='edge',label=legends[i])
		else:
			cumtmp = 0
			rects = stackvAX.barh(ylocations,np.asarray(stackvdata[i,:]),width,color=hcolors[legends[i]],linewidth=linewidth,alpha=alpha,align='edge',label=legends[i])
	stackvAX.legend(loc=0, fancybox=False,frameon=False,numpoints=1,handlelength=0.75)
	stackvAX.set_ylim(ylocations[0],ylocations[-1]+width)
	retain_x(stackvAX)
	if isinstance(CNV,pd.DataFrame):
		cnvcolors = ["b","r"]
		stackvAX = fig.add_subplot(GS[cnvpos,0])
		ngenes, nsamples= CNV.shape
		legends = []
		legendsnames = []
		samplenames = CNV.columns
		genenames = CNV.index
		cnvlist = gainloss.index
		cnvcutoff = [-1,1]
		htmp = {}
		for i in xrange(ngenes-1,-1,-1):
			if i <> ngenes-1:
				cumtmp = cumtmp + np.ones(len(samplenames))
				for j in xrange(len(cnvlist)):
					tmpidx = CNV.loc[genenames[i]] == cnvcutoff[j]
					if tmpidx.sum() >= 1:
						rects = stackvAX.bar(xlocations[tmpidx.values],ones[tmpidx.values]*0.9,width,color=cnvcolors[j],linewidth=0,alpha=0.8,bottom=cumtmp[tmpidx.values],align='edge')
						if cnvlist[j] not in htmp:
							legends.append(rects);legendsnames.append(cnvlist[j]);htmp[cnvlist[j]] = None
			else:
				cumtmp = 0
				for j in xrange(len(cnvlist)):
					tmpidx = CNV.loc[genenames[i]] == cnvcutoff[j]
					if tmpidx.sum() >= 1:
						rects = stackvAX.bar(xlocations[tmpidx.values],ones[tmpidx.values]*0.9,width,color=cnvcolors[j],linewidth=0,alpha=0.8,align='edge')
						if cnvlist[j] not in htmp:
							legends.append(rects);legendsnames.append(cnvlist[j]);htmp[cnvlist[j]] = None
		stackvAX.set_xlim(xlocations[0],xlocations[-1]+width)
		stackvAX.set_ylim(0,ngenes-0.1)
		stackvAX.set_yticks(np.arange(ngenes)+ 0.5)
		stackvAX.set_yticklabels(genenames[::-1])
		framey(stackvAX)
		ax = fig.add_subplot(GS[cnvpos+2,0])
		ax.legend(legends,legendsnames,loc=0, ncol=7,fancybox=False,frameon=False,numpoints=1,handlelength=0.75)
		clean_axis(ax)
		# plot gain loss
		stackvdata = gainloss.values
		ylocations = np.arange(ngenes)
		n,p = stackvdata.shape
		stackvAX = fig.add_subplot(GS[cnvpos,1])
		linewidth = 0
		alpha=0.8
		width=0.95
		legends = ["Loss","Gain"]
		for i in xrange(n):
			if i:
				cumtmp = cumtmp + np.asarray(stackvdata[i-1,:])
				rects = stackvAX.barh(ylocations,np.asarray(stackvdata[i,:]),width,color="r",linewidth=linewidth,alpha=alpha,left=cumtmp,align='edge',label=legends[i])
			else:
				cumtmp = 0
				rects = stackvAX.barh(ylocations,np.asarray(stackvdata[i,:]),width,color="b",linewidth=linewidth,alpha=alpha,align='edge',label=legends[i])
		stackvAX.legend(loc=0, fancybox=False,frameon=False,numpoints=1,handlelength=0.75)
		stackvAX.set_ylim(ylocations[0],ylocations[-1]+width)
		retain_x(stackvAX)

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

