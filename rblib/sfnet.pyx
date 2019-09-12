import sys
import numpy as np
from rblib import cluster
import pandas as pd
import h5py
from rblib.gwaspls import resampling2
from scipy import stats
from rblib import statplot
import os
# power call # default use 6, + or - need to be consider
# Often choosing beta=6 works well but in general we use the “scale free topology criterion” described in Zhang and Horvath 2005.


def rddm(datamatrix,anno,selected=5000):
	ret = None
	n,p = datamatrix.shape
	# idx = resampling2(np.ones(self.p),min(selectsamples,self.p))
	idx = resampling2(np.ones(p),min(selected,p))
	return datamatrix[:,idx],np.asarray(anno)[idx]


class SFnet(object):
	def __init__(self,anno,tmppath,power=6,pcut=0.05,signtype=0,minsize=20,threads=8,ncluster=20,outdir="./"):
		self.power = power
		self.signtype = signtype
		self.modules = None
		self.anno = anno
		self.p = len(anno) 
		self.minsize = minsize
		self.threads = threads
		self.ncluster = ncluster
		self.tmppath = tmppath
		self.pcut    = pcut		
		self.datafile = h5py.File('%s/data.hdf'%self.tmppath,'a')
		self.outdir = outdir
	def __writeCorr(self):
		p = self.p
		Xcorr = np.eye(p)
		for i in range(p-1):
			dataset = h5py.File('%s/tmpdata_%d.hdf'%(self.tmppath,i),'r')
			Xcorr[i,i+1:] = dataset["correlations"][:]
			Xcorr[i+1:,i] = dataset["correlations"][:]
			dataset.close()
		if 'corr' in self.datafile: del self.datafile['corr']
		self.datafile.create_dataset('corr', data = Xcorr)
		return 0
	def __writePvalue(self):
		p = self.p
		pvalues = np.zeros((p,p))
		for i in range(p-1):
			dataset = h5py.File('%s/tmpdata_%d.hdf'%(self.tmppath,i),'r')
			pvalues[i,i+1:] = dataset["pvalues"][:]
			pvalues[i+1:,i] = dataset["pvalues"][:]
			dataset.close()
		if 'pvalue' in self.datafile: del self.datafile['pvalue']
		self.datafile.create_dataset('pvalue', data = pvalues)
		return 0
	def __deltmpfile(self):
		p = self.p
		for i in range(p-1):
			os.remove('%s/tmpdata_%d.hdf'%(self.tmppath,i))
		return 0
	def initCorr(self):
		self.__writeCorr()
		self.__writePvalue()
		self.__deltmpfile()
		return 0

	def determine_power(self,selectsamples = 5000,nbins = 20):
		idx = resampling2(np.ones(self.p),min(selectsamples,self.p))	
		idx.sort()
		
		tmpcorr = self.datafile['corr'][idx][:,idx]
		if self.signtype == 0:
			tmpcorr = np.abs(tmpcorr)
		elif self.signtype == 1:
			tmpcorr = (1+tmpcorr) / 2
		elif self.signtype == 2:
			tmpcorr[tmpcorr<0] = 0
		powers  = np.arange(15) + 1
		rsquare = []
		for power in powers:
			tmpconnectivity = np.sum(tmpcorr ** power,axis=0) - 1
			density,binedge = np.histogram(tmpconnectivity,nbins,density=True)
			bins = (binedge[0:-1] + binedge[1:])/2
			Y = np.log10(density)[density>0]		
			X = np.log10(bins)[density>0]
			slope,intercept,rvalue,pvalue,stderr = stats.linregress(X,Y)
			rsquare.append(rvalue ** 2)
			statplot.plot_simple_lr(X,Y,"$log_{10}k$","$log_{10}p(k)$",figname_prefix=self.outdir+"/"+"lr_%d_power"%power)
		statplot.scatter2(powers,rsquare,"Power","R-square",addline=[[1,15],[0.9,0.9]],fig_prefix=self.outdir+"/"+"scale-free-criterion_cutoff",alpha=0.6)
		return 0

	def power_adjacency(self):
		sys.stderr.write("[Info] copy to memory\n")
		Xadj = self.datafile['corr'][:]
		sys.stderr.write("[Info] start power call preprocess\n")
		if self.signtype == 0:
			Xadj = np.abs(Xadj)
		elif self.signtype == 1:
			Xadj += 1
			Xadj /= 2.0
		elif self.signtype == 2:
			Xadj[Xadj < 0] = 0
		sys.stderr.write("[Info] start power call\n")
		Xadj **= self.power
		n,p = Xadj.shape
		sys.stderr.write("[Info] refresh Xadj\n")
		for i in range(n):
			Xadj[i,i] = 0 ### here has minus itself, so i == j, adj = 0, convenient for get connectivity use sum function
		sys.stderr.write("[Info] store Xadj\n")
		if 'adjacency' in self.datafile: del self.datafile['adjacency']
		self.datafile.create_dataset('adjacency', data = Xadj)
		return 0

	def callconnectivity(self):
		if 'connectivity' in self.datafile: del self.datafile['connectivity']
		connectivity = np.sum(self.datafile['adjacency'],axis=0) # -1 # to Minus itself
		self.datafile.create_dataset('connectivity', data = connectivity)
		return 0
	def dissTOM(self):
		"""
		test memory 
		"""
		Tom = np.zeros((self.p,self.p))
		atmp2 = np.zeros((self.p,1))
		for i in range(self.p):
			atmp1 = self.datafile['adjacency'][i,:]
			atmp2[:,0] = self.datafile['adjacency'][:,i]
			tmpmink = np.minimum(self.datafile['connectivity'],self.datafile['connectivity'][i])
			tmpTom = np.dot(self.datafile['adjacency'],atmp2)
			tmpTom =  (tmpTom[:,0] + atmp1) / (tmpmink+1-atmp1)
			tmpTom[i] = 1
			tmpTom = 1 - tmpTom
			Tom[i,:] = tmpTom
		#=============
		"""	
		Tom = np.dot(self.datafile['adjacency'],self.datafile['adjacency'])
		for i in range(self.p):
			atmp = self.datafile['adjacency'][i,:]
			tmpmink = np.minimum(self.datafile['connectivity'],self.datafile['connectivity'][i])
			x = Tom[i,:] + atmp
			print x.shape
			Tom[i,:] = (Tom[i,:] + atmp)/(tmpmink+1-atmp)
			Tom[i,i] = 1
		Tom *= -1 # to get disstom	
		Tom += 1 # Tom is 1-Tom
		print Tom == Tom1
		sys.exit(1)
		"""
		if 'disstom' in self.datafile: del self.datafile['disstom']
		self.datafile.create_dataset('disstom', data = Tom) # 
		return 0
	def clustermodule(self):
		Xdist = self.datafile['disstom'][:]
		ncluster = min(self.p-1,self.ncluster)
		disstomcluster = cluster.Cluster(Xdist=Xdist,threads=self.threads,ncluster=ncluster)
		clusterlabels = disstomcluster.hcluster()
		del Xdist
		clusternames = set(clusterlabels.tolist())
		self.modules = np.asarray(["Unknown",]*len(self.anno))
		subconnectivity = np.zeros(len(self.anno))
		for name in clusternames:
			idx = np.where(clusterlabels == name)
			idx = idx[0].tolist()
			if len(idx) < self.minsize:
				name = "others"
			name = str(name)
			self.modules[idx] = name 
			#tmpXadj = self.adjacency[idx,idx]
			tmpXadj = self.datafile['adjacency'][idx][:,idx]
			subconnectivity[idx] = np.sum(tmpXadj,axis=0)
		if 'subconnectivity'  in self.datafile: del self.datafile['subconnectivity']
		self.datafile.create_dataset('subconnectivity',data=subconnectivity)
		return 0
	def outputmodule(self,fileout="sfnet.out"): # output cut off for edge 
		# print connectivity
		f1 = open(self.outdir+"/"+fileout + ".nodes.tsv","w")
		f2 = open(self.outdir+"/"+fileout + ".edges.tsv","w")
		f1.write("#Gene\tModule_id\tInModule.connectivity\tOverall.connectivity\n")
		f2.write("#Gene1\tGene2\tcorr\tpvalue\tadjacency\n")
		p = len(self.anno)
		for i in range(p):
			name = self.anno[i]
			f1.write("%s\t%s\t%s\t%s\n"%(name,self.modules[i],self.datafile['subconnectivity'][i],self.datafile['connectivity'][i]))
		f1.close()
		
		for i in range(p-1):
			for j in range(i+1,p):
				tmpadj = self.datafile['adjacency'][i,j]
				if self.datafile['pvalue'][i,j] < self.pcut:
					f2.write("%s\t%s\t%.7f\t%.3e\t%.7f\n"%(self.anno[i],self.anno[j],self.datafile['corr'][i,j],self.datafile['pvalue'][i,j],tmpadj))
		return 0
	def close(self):
		self.datafile.close()
		return 0
