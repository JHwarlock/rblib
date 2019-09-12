import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import itertools
from rblib import mplconfig

from rklib.utils import us_sort
import numpy as np
matplotlib.rcParams['axes.facecolor'] = u"#FFFFFF"

COLOR = {
		'metastasis':'#00FFF2',
		'replased':'#FF00E7',
		'no':'#9467BD',
		'edge':'#777777',
		'startcodon':'#55FF55',
		'endcodon':'#FF5555',
		'novelcolor':'#467821',
		'cds':'#FF7D0B',
		'None':'#F0F0F0',
		}
GROUPCOLOR = {
		'metastasis':'blue',
		'relapsed':'red',
		'ln_metastasis':'#55FF55',
		'good':'#9467BD',
		}

MUTCOLOR = {
		"Frame_Shift_Del":None,
		"Frame_Shift_Ins":None,
		"In_Frame_Del":None,
		"In_Frame_Ins":None,
		"Missense_Mutation":None,
		"Nonsense_Mutation":None,
		"Nonstop_Mutation":None,
		"Silent":None,
		"Splice_Site":None,
		"Splice_Region":None,
		"Translation_Start_Site":None,
		}
MUTCOLOR = mplconfig.inscolor(sorted(MUTCOLOR.keys()),"Set3")
MUTCOLOR["None"] = "#C0C0C0"

GROUPCOLOR = mplconfig.inscolor(sorted(GROUPCOLOR.keys()),"gist_rainbow")

NORMALSTYLE = { 'ls':'solid',  'lw':0.5, 'weight':'bold',  'style':'normal',"alpha":0.5,'marker':'o','markersize':8,"arrWidth":0.6,"markeredgewidth":1.0}
ANNOSTYLE = { 'ls':'solid',  'lw':0.5, 'weight':'bold',  'style':'normal',"alpha":0.5,"arrWidth":0.7}
MUTSTYLE = {'ls':'solid',  'lw':1.5, 'weight':'bold',  'style':'normal',"alpha":0.5,'marker':None,'markersize':0}


arrWidth = 0.5
maxHeadLen = 150

from rblib.statplot import clean_axis


class ProtPlot(object):
	def __init__(self,genename,currentax = None,ax = None,fig = None):
		figwidth = 8
		figheigt = 12
		self.numtracks  = 1
		self.fig = plt.figure(figsize=(figwidth,figheigt),dpi=300) if fig == None else fig
		self.currentax  = 1 if currentax == None else currentax
		self.ax = ax
		self.genename = genename 
		self.sy = 0.7
		self.sy_step = 0.05
		self.window = 500.0
		self.bands  = 0
		self.legends = [] # franctions, name, class
		self.legendnames = []
	def __creatax(self):
		#assert self.currentax <= self.numtracks
		self.ax = self.fig.add_subplot(self.numtracks,1,self.currentax)
		self.ax.set_title(self.genename)
		self.currentax += 1
		return self.ax
	
	def __plotarrow(self,x,y,dx,dy=0.0,regiontype = "None",styles = None, color = None,ec_color=None):
		styles = NORMALSTYLE if styles == None else styles
		color  = COLOR if color == None else color
		headLength =  0#min(abs(dx)/2.0, maxHeadLen)
		if ec_color == None:
			ec_color = color[regiontype]
		patch = self.ax.arrow(x, y, dx, dy,fc=color[regiontype], ec=ec_color, ls=styles['ls'], lw=styles['lw'],width=styles["arrWidth"],head_width=styles["arrWidth"],head_length=headLength,shape='full',length_includes_head=True)
		return patch
	def __scatter(self,x1,x2,y1,y2,muttype="None",grouptype = "None",styles = None , mutcolor = None,groupcolor=None,addline = 0):
		styles = NORMALSTYLE if styles == None else styles
		mutcolor  = COLOR if mutcolor == None else mutcolor
		groupcolor= COLOR if groupcolor==None else groupcolor
		if not addline:
			self.ax.plot([x1,x2],[y1,y2],color="black",ls=styles['ls'],lw=1.0)
			scat1 = self.ax.plot([x2,],[y2,],color=mutcolor[muttype],marker=styles['marker'],markersize=styles['markersize']-2,markeredgecolor='None',lw=0,ls='None')
		scat2 = self.ax.plot([x2,],[y2,],color='None',marker=styles['marker'],markersize=styles['markersize']+1,markeredgecolor=groupcolor[grouptype],markeredgewidth=styles['markeredgewidth'],lw=0,ls='None')
		if muttype not in self.legendnames and not addline:
			self.legends.append([scat1[0],muttype,"Mutation"])
			self.legendnames.append(muttype)
		if grouptype not in self.legendnames:
			self.legends.append([scat2[0],grouptype,"Group"])
			self.legendnames.append(grouptype)
		return 0
	
	def __text(self,x,y,text,rotation=90):
		self.ax.text(x,y,text,rotation=rotation,horizontalalignment="center",verticalalignment="bottom",fontsize=10)
		return 0

	def __linetext(self,x1,x2,y1=0.35,y2=0.7,regiontype = "None",styles = None, color = None):
		styles = NORMALSTYLE if styles == None else styles
		color  = COLOR if color == None else color
		self.ax.plot([x1,x2],[y1,y2],color=color[regiontype],ls=styles['ls'],lw=1.0)
		#self.ax.plot([x1,x2],[y2,y2+0.15],color=color[regiontype],ls=styles['ls'],lw=styles['lw'])
		return 0
	def creatax(self):
		self.__creatax()
		return 0

	def plotwhole(self,protlen):
		self.__plotarrow(0,0,self.window,0.0,regiontype="None",ec_color="black")
		self.bands = protlen/self.window
		print(self.bands)
		self.ax.set_xlim(-50,self.window+50)
		return 0

	def plotanno(self,annoregions): # [start,end ,name],
		h = {}
		for start,end,name in annoregions:
			if name not in h:h[name] = []
			h[name].append([start,end])
		color = mplconfig.inscolor(h.keys(),"Paired")
		for name in h:
			for start,end in h[name]:
				patch = self.__plotarrow(start/self.bands,0.0,(end-start)/self.bands,0.0,regiontype = name,styles = ANNOSTYLE , color=color)
				if name not in self.legendnames:
					self.legends.append([patch,name,"Anno"])
					self.legendnames.append(name)
		return 0

	def plotmut(self,mutationlist,groupcolor = None,mutcolor = None):# [mutppos,patr,group,muttype] # 2317 p.R2317Q,metas,nonsence
		h = {}
		for mutppos,patr,group,muttype in mutationlist:
			if mutppos not in h: h[mutppos] = {}
			if patr not in h[mutppos]: h[mutppos][patr] = []
			h[mutppos][patr].append([group,muttype])

		for mutppos in h:
			numpatrs = len(h[mutppos].keys())
			bands = 14 * (numpatrs - 1)
			posarr = np.linspace(mutppos/self.bands - bands/2,mutppos/self.bands + bands/2,numpatrs)
			idx = 0
			self.__linetext(mutppos/self.bands,mutppos/self.bands,0.35,0.7,regiontype = "None",styles = None, color = {'None':'black'})
			for patr in h[mutppos]:
				mutppostmp = posarr[idx]
				attrs = us_sort(h[mutppos][patr],0)
				offset = 0.0
				mutppostmp1 = mutppos/self.bands
				mutppostmp2 = mutppostmp
				samplesidx = 0
				ytmp = 0
				for group,muttype in attrs:
					ytmp = self.sy + offset
					self.__scatter(mutppostmp1,mutppostmp2,ytmp,ytmp+0.2,muttype=muttype,grouptype=group,styles = None, mutcolor = mutcolor,groupcolor = groupcolor, addline=samplesidx)
					mutppostmp1 = mutppostmp
					offset += 0.2
					samplesidx += 1
				ytmp = self.sy + offset
				self.__text(mutppostmp,ytmp+0.14,".".join(patr.split(".")[1:]),rotation=90)
				idx += 1
		return 0
	def save(self,figprefix="testplot"):
		clean_axis(self.ax)
		self.ax.set_ylim(-10,6)
		#process legends
		#self.ax.legend(self.legends,self.legendnames,loc=0,fancybox=False,frameon=False,numpoints=1,handlelength=0.75)
		legends_sort = us_sort(self.legends,2)
		tmplg = []
		tmplgname = []
		for rect,name,cate in legends_sort:
			tmplg.append(rect)
			tmplgname.append(name)
		self.ax.legend(tmplg,tmplgname,loc=0,fancybox=False,frameon=False,numpoints=1,handlelength=0.75)
		self.fig.tight_layout()
		plt.savefig(figprefix+".png",format='png',dpi=300)
		plt.savefig(figprefix+".svg",format='svg',dpi=300)
		#plt.savefig(figprefix+".pdf",format='pdf',dpi=300)
		sys.stderr.write("[INFO] png and svg has been saved with prefix '%s'\n"%figprefix)
		plt.clf();plt.close();
		return 0

if __name__ == "__main__":
	hpos= {
			1399:["metastasis","metastasis","replased","replased","no","no"],
			1032:["metastasis","metastasis"],
			}
	mutationlist = [[1032,"p.A1032K","metastasis","Missense_Mutation"],[1399,"p.R1399N","metastasis","Nonsense_Mutation",],[1399,"p.R1399N","relapsed","Silent"],[1399,"p.R1399*","metastasis","Frame_Shift_Del"]]
	#mutationlist = [[1032,"p.A1032K","metastasis","None"],[1399,"p.R1399N","metastasis","None",],[1399,"p.R1399N","replased","None"]]
	protlen = 2570
	annoregions = [[334,712,"A"],[1000,1621,"B"]]
	A = ProtPlot("test")
	A.creatax()
	A.plotwhole(protlen)
	A.plotanno(annoregions)
	A.plotmut(mutationlist,groupcolor=GROUPCOLOR,mutcolor=MUTCOLOR)
	A.save()



