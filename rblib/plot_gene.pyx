# -*- coding: UTF-8 -*-
import sys
import numpy as np
import pysam
import bamio
import scipy as sp
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm
import statplot
import matplotlib.cm as cm
import ssbamparse
from  rklib.utils import us_sort
from seqio import merge_region

def format_pos(a):
	if a > 1000000:
		return "%.3fMb"%(float(a)/1000000)
	elif a > 1000:
		return "%.3fKb"%(float(a)/1000)
	else:
		return "%dbp"%a

class Exon_struct_plot(object):
	def __init__(self,samplenum=0,biotypes=[],dpi=300,sncolors=None):
		self.snnum = samplenum ## 样本的数目，图的数目是 样本数+1 ， 1为基因结构图, biotypes 为总共要画的这个区域内的基因的类型。
		self.fig = plt.figure(figsize=(12,self.snnum*3+3),dpi=300)
		self.subplotnum = "%d1"%(self.snnum+1)
		self.subplotnum1 = self.snnum+1
		self.subplotnum2 = 1
		self.subplotnum3 = 0
		self.ax_gene = self.fig.add_subplot(self.subplotnum1,self.subplotnum2,1)
		self.biotypecolors = {}
		self.offset = 0
		self.tids = []
		setbiotypes = list(set(biotypes))
		ret_colors = statplot.color_grad(len(setbiotypes),cm.gist_rainbow)
		for i in xrange(len(setbiotypes)):
			self.biotypecolors[setbiotypes[i]] = ret_colors[i]
		self.allplotexons = {}
		self.startx = 0
		self.endx   = 0
		self.chromx = ""
	def _plot_coor(self,chrom,startx,endx,frac=5):
		region = np.linspace(startx,endx,frac).tolist()
		strregion = map(format_pos,region)
		self.ax_gene.plot(region,[self.offset,]*frac,'k-')
		self.ax_gene.set_xticks(region)
		self.ax_gene.set_xticklabels(strregion,rotation=0)
		self.ax_gene.set_xlim(startx,endx)
		self.ax_gene.set_ylim(bottom=0)
		self.ax_gene.set_xlabel(chrom)
		self.ax_gene.spines['top'].set_visible(False)
		self.ax_gene.spines['bottom'].set_visible(False)
		self.ax_gene.spines['right'].set_visible(False)
		self.ax_gene.spines['left'].set_visible(False)
		self.ax_gene.get_xaxis().tick_bottom()
		self.ax_gene.get_yaxis().set_ticks([])
		self.startx = startx
		self.endx   = endx
		self.chromx = chrom
		#print self.chromx,self.startx,self.endx
	def _ax_subplot(self,ax,chrom,startx,endx,frac=5):
		region = np.linspace(startx,endx,frac).tolist()
		strregion = map(format_pos,region)
		ax.plot(region,[0,]*frac,'k-')
		ax.set_xticks(region)
		ax.set_xticklabels(strregion,rotation=0)
		ax.set_xlim(startx,endx)
		#print chrom,startx,endx
	
	def _add_genes_struct(self,chrom,startx,endx,plot_symbols_exons,gn):# 添加一个基因所有的转录本信息。可添加多个基因，以画在一个图内 
		# chrom id 
		# + , - 
		# 画图的位置，start 和 end ， 也标记了xlim
		# cstart 是 biotype = prot coding ？ or others start pos
		# gn -> tid,tname,chrom_t,start_t,end_t,strand,exons,biotype = plot_symbols_exons[i]  
		# start_t  :  coding start ,  0 or  None 表示没有
		# end_t    :  coding end ，   
		#self.ax_gene = self.fig.add_subplot(self.subplotnum+str(idx)
		#plotted = []
		#hplotted = {}
		for i in xrange(len(plot_symbols_exons)):
			#tmpbiotype = biotype[i]
			#tcolor = self.biotypecolors[i]
			tid,tname,chrom_t,start_t,end_t,strand,exons,biotype = plot_symbols_exons[i]
			tcolor = self.biotypecolors[biotype]
			#print transname,chrom_t,start_t,end_t,strand,exons
			#print plot_symbols_exons[i]
			assert chrom_t == chrom
			try:
				#print startx <= exons[0][0]  and  exons[-1][-1] <= endx
				#print exons,tid,tname,chrom_t,start_t,end_t,strand,biotype,gn
				#print startx,endx
				assert startx <= exons[0][0]  and  exons[-1][-1] <= endx
			except:
				startx = min(startx, exons[0][0]); endx = max(endx, exons[-1][-1])
				self.startx = startx
				self.endx   = endx
				print >> sys.stderr, '[WARN]: Original position infos has been updated by: chrom -> {} start -> {} end -> {}'.format(chrom, startx, endx)
				#continue
			#print gn,tid
			self.tids.append(gn+"|"+tid)
			t = np.linspace(startx,endx,7)[1:-1]
			self.offset += 1
			tmpyoffset = np.ones(len(t))*self.offset
			#print tmpyoffset
			if chrom_t not in self.allplotexons:
				self.allplotexons[chrom_t] = []
			tmps = []

			for pos in exons:
				starttmp,endtmp = pos
				#if endtmp < startx:continue
				#if starttmp > endx:continue
				#starttmp = max(starttmp,startx)
				#endtmp = min(endtmp,startx)
				self.allplotexons[chrom_t].append([starttmp,endtmp])
				tmps.append([starttmp,endtmp-starttmp])
				#self.ax_gene.plot((starttmp,endtmp),(self.offset,self.offset),"-",color=tcolor,lw = 4,alpha=0.8)
			#self.allplotexons[chrom_t].extend(exons)
			self.ax_gene.broken_barh(tmps,(self.offset-0.15,0.3),facecolors = tcolor,alpha=0.6,lw=0)
			if start_t:
				self.ax_gene.plot((start_t,start_t+3),(self.offset,self.offset),"k|-",markersize = 8)
			if end_t:
				self.ax_gene.plot((end_t,end_t+3),(self.offset,self.offset),"k|-",markersize = 8)
			if strand == "+":
				self.ax_gene.plot(t,tmpyoffset,"k>--",markersize=4,lw=0.1)
			elif strand == "-":
				self.ax_gene.plot(t,tmpyoffset,"k<--",markersize=4,lw=0.1)
			else:
				self.ax_gene.plot(t,tmpyoffset,"k--",markersize=4,lw=0.1)
		self.ax_gene.set_yticks(np.arange(1,self.offset+1,1))
		self.ax_gene.yaxis.set_ticks_position('left')
		yticklabels = self.ax_gene.set_yticklabels(self.tids)
		for l in self.ax_gene.get_yticklines():
			l.set_markersize(0)
		for yticklabel in yticklabels:
			yticklabel.set_fontsize(8)
		self.ax_gene.set_ylim(0,self.offset+1)
		self.ax_gene.grid(True)
		#for i,j,k,l,m,n in plotted:
		#	self.ax_gene.broken_barh(i,j,facecolors = k,alpha=l,lw=m,label=n)
		#self.ax_gene.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, )
		self.fig.tight_layout()
		#self.offset += 2
		return 0

	def _sortmerge_plotexon(self):# before plot bams , first to sort and  merget regions for exons, to plot one read at once. 
		for chrom_t in self.allplotexons:
			self.allplotexons[chrom_t] = merge_region(us_sort(self.allplotexons[chrom_t],0,1))[:]

	def plot_sbam(self,sns,sbams,sslib=0,plotnum=2,plotdata="counts"):# must be sorted bams with index #
		# plotnum 代表了subplot 的开始的编号
		assert len(sns) == len(sbams)
		if sslib >= 1: 
			parsefun = ssbamparse.parse_strand_specificbam
		else:
			parsefun = ssbamparse.parse_unstrand_bam
		max_plus = 0
		max_minus = 0
		axs = []
		self._sortmerge_plotexon()
		#print self.subplotnum
		#print str(0+plotnum)
		if plotdata == "counts":
			for i in xrange(len(sns)):
				axs.append(self.fig.add_subplot(self.subplotnum1,self.subplotnum2,i+plotnum))
				snname = sns[i]
				sbam  = sbams[i]
				fbamfile1 = pysam.Samfile(sbam,"rb")
				fbamfile2 = pysam.Samfile(sbam,"rb")
				for chrom_t in self.allplotexons:
					exons = self.allplotexons[chrom_t]
					for start,end in exons:
						tmpoffset_plus = 0;tmpoffset_minus = 0;
						for ret in parsefun(fbamfile1,fbamfile2,chrom_t,start,end,sslib=sslib):
							read1blocks,read2blocks,inter,strand = ret
							#print read1blocks
							#print read2blocks
							if strand == "." or strand == "+":
								tmpoffset_plus += 1
								tmpoffset = tmpoffset_plus
							else:
								tmpoffset_minus -= 1
								tmpoffset = tmpoffset_minus
							self.__plotblocks(read1blocks,"r",axs[i],tmpoffset)
							self.__plotblocks(read2blocks,"b",axs[i],tmpoffset)
							self.__plot_inter(inter,axs[i],tmpoffset)
							
						max_plus = np.max((tmpoffset_plus,max_plus))
						max_minus = np.min((tmpoffset_minus,max_minus))
						#axs[i].plot(t,[tmpoffset,tmpoffset],"-",color="r",lw=5,alpha=0.8)
				fbamfile1.close()
				fbamfile2.close()
				if sslib >= 1:
					axs[i].set_ylabel("Read counts on different strands (%s)"%snname)
				else:
					axs[i].set_ylabel("Read counts on genome (%s)"%snname)
		elif plotdata == "TPM":# tmp for each pos
			for i in xrange(len(sns)):
				#axs.append(self.fig.add_subplot(self.subplotnum+str(i+plotnum)))
				axs.append(self.fig.add_subplot(self.subplotnum1,self.subplotnum2,i+plotnum))
				snname = sns[i];sbam  = sbams[i]
				fbamfile1 = pysam.Samfile(sbam,"rb")
				fbamfile2 = pysam.Samfile(sbam,"rb")
				for chrom_t in self.allplotexons:
					exons = self.allplotexons[chrom_t]
					for start,end in exons:
						t = np.linspace(start,end,int((end-start)/10))
						tmp_tpms_plus = [];
						tmp_tpms_minus = [];
						for pos in t:
							count_plus,count_minus,tpm_plus,tpm_minus  =  ssbamparse.getcovpos(chrom_t,pos,pos,fbamfile1,fbamfile2,sslib=sslib)
							max_plus = np.max((tpm_plus,max_plus))
							max_minus= np.max((max_minus,tpm_minus))
							tmp_tpms_plus.append(tpm_plus)
							tmp_tpms_minus.append(tpm_minus)
						self.__plottpm(t,tmp_tpms_plus,axs[i],"r")
						#self.__plottpm(t,-1*np.asarray(tmp_tpms_minus),axs[i],"b")
				
				if sslib >= 1:
					axs[i].set_ylabel("Raw scaled counts on different strands (%s)"%snname)
				else:
					axs[i].set_ylabel("Raw scaled counts on genome (%s)"%snname)
			max_minus = -1 * max_minus
		for ax in axs:
			if sslib <1:
				ax.set_ylim(0,max_plus)	
			elif sslib >=1:
				ax.set_ylim(max_minus,max_plus)
			self._ax_subplot(ax,self.chromx,self.startx,self.endx,frac=5)
				
	def __plot_inter(self,inter,ax,offset,lw=0.3):
		if inter:
			start = inter[0]
			end   = inter[1]
			t = np.linspace(start,end,6)
			ax.plot(t,np.ones(6)*offset,"--",color='k',lw=0.2)

	def __plotblocks(self,readblocks,color,ax,offset,lw=0,alpha=0.6):
		if readblocks:
			tmp = []
			for tmpstart,tmpend in readblocks:
				tmp.append([tmpstart,tmpend-tmpstart])
				#tmpyoffset = np.ones(2)*offset
			#broken_barh(tmps,(self.offset-0.15,0.3),facecolors = tcolor,alpha=0.6,lw=0)
			ax.broken_barh(tmp,(offset-0.45,0.9),facecolors=color,lw=lw,alpha=alpha)
			if len(readblocks) >=2:
				for i in xrange(len(readblocks)-1):
					end = readblocks[i+1][0]
					start = readblocks[i][1]
					t = np.linspace(start,end,2)
					ax.plot(t,np.ones(2)*offset,"-",color='k',lw=0.2)
	
	def __plottpm(self,x,y,ax,color):
		# plt.fill(x, y, 'r')
		#print [x[0]-1,]+x.tolist()+[x[-1]+1,]
		#print [0,]+y+[0,]
		if len(x)>0 and len(y)>0:
			ax.fill([x[0]-1,]+x.tolist()+[x[-1]+1,],[0,]+y+[0,],color)

	def save(self,figprefix):
		self.fig.savefig(figprefix+".png",format='png',dpi=300)
		self.fig.savefig(figprefix+".svg",format='svg',dpi=300)
		self.fig.clf()
		plt.close()



if __name__ == "__main__":
	biotypes = ["coding","lnc"]
	a = Exon_struct_plot(biotypes=biotypes)
	chrom = "chr1"
	a._plot_coor(chrom,10010,15000)
	startx = 10010 # plot start pos
	endx = 15000
	#transname,chrom_t,start_t,end_t,strand,exons,biotype 
	plot_symbols_exons = [["AT000000000001","ATname1","chr1",10040,14750,"-",[[10020,14820],],"coding"],["AT02","ATname2","chr1",10045,14802,"-",[[10020,12100],[13141,14890]],"lnc"]]
	a._add_genes_struct(chrom,10010,15000,plot_symbols_exons,"A")
	a.save("testxxx")


