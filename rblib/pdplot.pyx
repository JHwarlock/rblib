#import pandas as pd
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
import numpy as np
mpl.rcParams.update({'font.size': 10}) 
#mpl.rcParams['legend.fontsize'] = 9
#mpl.style.use('ggplot')
import mplconfig

def gbplot_pie(fractions, #values for the wedges
	labels, #labels for the wedges
    title = None, #title of the pie chart
    cm_name = 'Paired', #name of the matplotlib colormap to use
    autopct = '%1.2f%%', #format the value text on each pie wedge
    labeldistance = 1.05, #where to place wedge labels in relation to pie wedges
	shadow = False, #shadow around the pie
    startangle = 90, #rotate 90 degrees to start the top of the data set on the top of the pie
	edgecolor = 'w', #color of pie wedge edges
    explode = None,
	width = 8, #width of the figure in inches
    height = 8, #height of the figure in inches
    grouping_threshold = None, #group all wedges below this value into one 'all others' wedge
    grouping_label = None,
	fig_prefix="test",
	scale = 1,
	outlegend= 1
	):
	if not grouping_threshold==None:
		if grouping_label == None:
			grouping_label = 'Others'
		# make to fractions ratio
		if scale:
			fractions_ratio = np.float64(fractions) / np.sum(fractions)
		else:
			fractions_ratio = np.float64(fractions)
		#print fractions_ratio
		row_mask = fractions_ratio > grouping_threshold# 0.05 
		meets_threshold = np.asarray(fractions)[row_mask].tolist()
		labels_threshold =np.asarray(labels)[row_mask].tolist()
		all_others = np.asarray(fractions)[~row_mask].sum()
		meets_threshold.append(all_others)
		labels_threshold.append(grouping_label)
	else:
		meets_threshold = fractions
		labels_threshold = labels
	num_of_colors = len(meets_threshold)
	colors = mplconfig.styles(num_of_colors)[0]
	fig = plt.figure(figsize=(width,height),dpi=300)
	ax = fig.add_subplot(111)
	#patches, texts

	plotlabels = []
	if outlegend:
		count = 1
		for i in xrange(len(labels_threshold)):
			plotlabels.append(str(count))
			count += 1
	else:
		plotlabels = labels_threshold

	wedges = ax.pie(meets_threshold,
			explode = explode,
			labels = plotlabels, 
			labeldistance = labeldistance,
			autopct = autopct,
			colors = colors,
			shadow = shadow, 
			startangle = startangle)
	for wedge in wedges[0]:
		wedge.set_edgecolor(edgecolor)
	
	if outlegend:
		tmplabels = []
		total = sum(meets_threshold)
		count = 1
		for i in xrange(len(labels_threshold)):
			fraction_tmp = float(meets_threshold[i])/total*100
			tmplabels.append((str(count)+": "+labels_threshold[i]+"("+ autopct+")")%fraction_tmp)
			count += 1
		lgd = ax.legend(tmplabels,bbox_to_anchor=(1.03, 1),loc=0,borderaxespad=0,numpoints=1,fontsize=9)
		art = []
		art.append(lgd)
	if title:
		ax.set_title(title)
	print autopct
	print meets_threshold
	plt.tight_layout()
	if outlegend:
		plt.savefig(fig_prefix+"_pie.png",format='png',additional_artists=art,bbox_inches="tight",dpi=300)
		plt.savefig(fig_prefix+"_pie.svg",format='svg',additional_artists=art,bbox_inches="tight",dpi=300)
	else:
		plt.savefig(fig_prefix+"_pie.png",format='png',dpi=300)
		plt.savefig(fig_prefix+"_pie.svg",format='svg',dpi=300)
	plt.clf()
	plt.close()
	return 0

if __name__ == "__main__":
	data = [35,120,123,41,73,21,1,8]
	#make 1,2,3,4,5,6....
	# add legends
	labels = ["Adadsad","Bdadadadad","Cdasdadad dsadasd ada ed","Ddasdad  dasdad","E","F","G","H"]
	gbplot_pie(data,labels,grouping_threshold = 0.05 ,grouping_label= "all others",outlegend=0)
