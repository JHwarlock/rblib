#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from rklib import compress
from rblib import mplconfig
import os
import numpy as np
import scipy as sp
from scipy import stats
import sys
import time
import ctypes
import itertools

def check_vecnan(nparr):
	idx = ~np.isnan(nparr)
	if np.sum(idx) >=1:
		return idx
	else:
		return None

import pandas as pd
class Pdsampleinfo(object):
	def __init__(self):
		self.snnum = 0
		self.df = None
		self.sns= []
	
	def parse_sampleinfo(self,sampleinfo,phenotype='category'):
		fh = open(sampleinfo,"r")
		header = None
		data = []
		for line in fh:
			if line.startswith("#"):
				if line.startswith("##"):continue
				else:
					header = line.rstrip("\n").lstrip("#").split("\t");continue;
			arr = line.rstrip(os.linesep).split("\t")
			data.append(arr)
			self.sns.append(arr[0])
		fh.close()
		self.df = pd.DataFrame(np.asarray(data),index=self.sns,columns=header)
		if phenotype == "category":
			self.df.iloc[:,3].astype("category")
			self.df.iloc[:,4].astype("category") # df["XXX"].cat.categories
		elif phenotype == "quantificat":
			self.df.iloc[:,3:5] = np.float64(self.df.iloc[:,3:5])
		else:
			sys.stderr.write("[ERROR] Unknown phenotype: '%s'%s"%(phenotype,os.linesep))
			return 1
		return 0

class SampleInfo(object):
	def __init__(self):
		#SN Filenames	Samplename  Classidx Classname
		#S1	xx_1.fq.gz,xx_2.fq.gz	S1sample	1	CK
		#S2 xx_1.fq.gz,xx_2.fq.gz   S2sample    1   CK
		self.samplenum = 0 # record 4
		
		self.classlabels = []    # record [CK,CK,MT,MT]   # 按读取的顺序
		self.uniqclasslabel = [] # record [CK,MT]         # 按读取的顺序
		
		self.samplecolors = []   # 用style 函数产生的 sample color 
		self.classcolors = []    # 用style 函数产生的 class  color
		self.samplelines = []    # 对应的线型
		self.classlines = []     #
		self.samplemarkers = []  # 对应的marker  'o','^','v','+'
		self.classmarkers = []   # 对应的marker 'o','o','^','^'
		
		self.uniqcolor = []      # 内置
		
		self.classnums =[]       # record [1,1,2,2,]  class index id 按顺序
		self.uniqclassnum = []   # record [1,2]
		
		self.sns = []            # 第一列
		self.samplenames=[]      # 第三列
		self.traits = []         # 第四列
		self.uniqline = []       # 内置 
		self.uniqmarker = []     # 内置
		
		self.files = []          # 记录文件["SN///filename",]
		self.hfiles = {}         # 记录hfiles["SN"] => [filename,filename]
		self.hidx = {}           # 记录hidx["CK"] => [0,1] 即某组样本的index 编号
		self.hclassidxsn = {}    # 记录hclassidxsn["CK"] => [sn,sn]
		self.classids = []       # 记录 nparray [1,1,2,2]
		self.hclassid2classname = {} # 记录 hclassid2classname["1"] => "CK"
	
	def parse_sampleinfo(self,sampleinfo,phenotype='category'):
		fh = open(sampleinfo,"r")
		classnumOrtraits = []
		for line in fh:
			if line.startswith("#") or line.startswith("\n") or line.startswith(" ") or line.startswith("\t"):continue
			nameid,filedetail,samplename,trait,other = line.rstrip("\n").split("\t",4)
			
			self.sns.append(nameid)
			files = filedetail.rstrip(",").split(",")
			self.hfiles[nameid] = []
			for filename in files:
				self.files.append(nameid+"///"+filename)
				self.hfiles[nameid].append(filename)
			classname = other.split("\t")[0]
			if classname not in self.hclassidxsn:
				self.hclassidxsn[classname] = []
			self.hclassidxsn[classname].append(nameid)
			self.hclassid2classname[trait] = classname
			self.samplenum += 1
			self.classlabels.append(classname)
			if classname in self.uniqclasslabel:pass
			else:self.uniqclasslabel.append(classname)
			classnumOrtraits.append(trait)
			self.samplenames.append(samplename)
		fh.close()
		## use sample number to get iteration colors, markers and lines
		self.uniqcolor,self.uniqline,self.uniqmarker = mplconfig.styles(len(self.uniqclasslabel))
		self.samplecolors,self.samplelines,self.samplemarkers = mplconfig.styles(self.samplenum)
		#self.uniqclasslabel = list(set(self.classlabels))
		self.uniqclassnum = list(range(len(self.uniqclasslabel)))
		h={}
		tmpclasslabels = np.asarray(self.classlabels)
		for i in range(len(self.uniqclasslabel)):
			tmplabel = self.uniqclasslabel[i]
			self.hidx[tmplabel] = tmpclasslabels == tmplabel
			h[tmplabel] = i
		for classlabel in self.classlabels:
			self.classnums.append(h[classlabel])
			self.classcolors = [self.uniqcolor[i] for i in self.classnums]
			self.classmarkers = [self.uniqmarker[i] for i in self.classnums]
			self.classlines  = [self.uniqline[i] for i in self.classnums]
		#print self.classcolors
		#print self.classmarkers
		#print self.classlines
		
		if phenotype == "category":
			self.traits = np.transpose(np.asmatrix(np.float64(classnumOrtraits[:])))
			self.classids = np.float64(classnumOrtraits[:])
		elif phenotype == "quantificat":
			self.traits = np.transpose(np.asmatrix(np.float64(classnumOrtraits[:]))) ## n X 1 matrix
		else:return 1
		#print self.traits
		return 0
class Pdmatrix(object):
	def __init__(self):
		self.p = 0
		self.n = 0
		"""
		217         #geneid ID2 SN1 SN2 SN3 SN4
		219         geneB   trB 2.123   123.1   21313.  12312.      // note: tab sep
		"""
		self.df = None
		self.anno = []    # ["genaeA|trA","geneB|trB"] ,
		#self.annosep = [] # ["genaeA|trA","geneB|trB"] 
		self.annodf = None
	def parse_matrix_anno(self,fmatrixanno,cutoff=-np.inf,percent=0.5,addtolog=0.000,log2tr=0,missvalue=np.nan):
		f = compress.cfread(fmatrixanno,"r")
		t0 = time.time()
		sys.stderr.write('[INFO] Start to Build data ...%s'%os.linesep)
		header = None
		for line in f:
			if line.startswith("#"):
				if line.startswith("##"):continue
				else:
					header = line.strip("#").rstrip("\n").split("\t");continue
			else:
				arr = line.rstrip("\n").split("\t")
				self.n = len(arr[2:])
				break
		f.seek(0)
		t0 = time.time()
		num = int(self.n * percent)
		sys.stderr.write("[INFO] parametre for cutting sample number is: %d\n"%num)
		self.p = 0
		for line in f:
			if line.startswith("#"):continue
			self.p += 1
		f.seek(0)
		data = np.zeros((self.n,self.p))
		realp = 0 
		filterp = 0
		anno1 = []
		anno2 = []
		for line in f:
			if line.startswith("#"):continue
			arr = line.rstrip("\n").rstrip().split("\t")
			try:
				tmpdata = np.float64(arr[2:])
			except:
				sys.stderr.write("[ERROR] some data could not transform to float%s"%os.linesep)
				sys.stderr.write("[ERROR] %s%s"%(line,os.linesep))
				return 1
			if len(tmpdata) != self.n:
				sys.stderr.write("[ERROR] number of each line is not the same%s"%os.linesep)
				sys.stderr.write("[ERROR] %s%s"%(line,os.linesep))
				return 1
			if self.n >=2:
				if np.nanstd(tmpdata,ddof=1) <=0:
					sys.stderr.write("[INFO] data: %s was filtered,no variation \n"%(arr[0]+": "+arr[1]))
					filterp += 1
					continue
				if np.sum(np.isnan(tmpdata)) > num:
					sys.stderr.write("[WARN] data: %s was filtered,too many NANs \n"%(arr[0]+": "+arr[1])) 
					filterp += 1 
					continue
				if np.sum(tmpdata == missvalue) > num:
					sys.stderr.write("[WARN] data: %s was filtered,too many Missing Values \n"%(arr[0]+": "+arr[1]))
					filterp += 1
					continue
				if np.sum(np.isnan(tmpdata)) + np.sum(tmpdata[~np.isnan(tmpdata)] < cutoff) > num:
					sys.stderr.write("[WARN] data: %s was filtered, too many exprs lower than noise \n"%(arr[0]+": "+arr[1]))
					filterp += 1
					continue
				if len(set(arr[2:])) <= 1:
					sys.stderr.write("[WARN] data: %s was filtered, because of no variation\n"%(arr[0]+": "+arr[1]))
					filterp += 1
					continue
			if log2tr:
				tmpdata = np.log2(tmpdata+addtolog)
			realp += 1
			if realp % 100000 == 0:
				sys.stderr.write("[INFO] parsed '%d' PASSED Data\n"%realp)
			data[:,realp-1] = tmpdata
			self.anno.append(arr[0] + "|" + arr[1])  
			#self.annosep.append(arr[0] + "|" + arr[1])
			anno1.append(arr[0])  
			anno2.append(arr[1])
		f.close()
		sys.stderr.write("[INFO] summary, filter numbers: %d\n"%filterp)
		sys.stderr.write("[INFO] summary, real numbers: %d\n"%realp)
		self.p = realp
		self.df = pd.DataFrame(data[:,0:realp],index=header[2:],columns=self.anno)
		assert len(self.anno) == self.p
		sys.stderr.write('[INFO] Data Built done! cost %.2fs\n'%(time.time()-t0))
		self.annodf = pd.DataFrame({
			'anno1':anno1,
			'anno2':anno2,
			})
		return 0
	
class MatrixAnno(object):
	##np.asarray(a[:,0].T)[0].tolist()
	def __init__(self):
		"""
		# file format:
		#geneid	ID2	SN1	SN2	SN3	SN4
		genaeA	trA	1.73	2.56	7.31	8.991
		geneB	trB	2.123	123.1	21313.	12312.      // note: tab sep
		"""
		self.p = 0 # p record: variate numbers
		self.n = 0 # n record: sample numbers 
		self.data  = None # np.matrix => n X p matrix
		self.anno = []    # ["genaeA\ttrA","geneB\ttrB"]
		self.annosep = [] # ["genaeA|trA","geneB|trB"] 
		self.anno1 = []   # ["genaeA","geneB"]
		self.anno2 = []   # ["trA","trB"]
	def parse_matrix_anno(self,fmatrixanno,cutoff=-10000000.0,percent=0.5,addtolog=0.001,log2tr=0):
		fh = compress.gz_file(fmatrixanno,"r") # -np.inf
		t0 = time.time()	
		sys.stderr.write('[INFO] Start to Build data ...\n')
		for line in fh:
			if line.startswith("#") or line.startswith("\n") or line.startswith(" ") or line.startswith("\t"):
				continue
			else:
				#arr = line.rstrip("\n").split("\t")
				arr = line.rstrip("\n").split("\t")
				self.n = len(arr[2:])
				break
		fh.seek(0)
		t0 = time.time()
		num = int(self.n * percent)
		self.p = 0
		for line in fh:
			if line.startswith("#") or line.startswith("\n") or line.startswith(" ") or line.startswith("\t"):continue
			else:
				self.p += 1
		fh.seek(0)
		self.data = np.zeros((self.n,self.p))
		realp = 0
		filterp = 0
		for line in fh:
			if line.startswith("#") or line.startswith("\n") or line.startswith(" ") or line.startswith("\t"):continue
			else:
				arr = line.rstrip("\n").rstrip().split("\t")
				try:
					tmpdata = np.float64(arr[2:])
				except:
					sys.stderr.write("[ERROR] %s"%line)
					sys.stderr.write("[ERROR] n is not same as exprsnums\n")
					return 1
				if self.n >=2:
					if np.nanstd(tmpdata,ddof=1) <=0:
						sys.stderr.write("[INFO] data: %s was filtered, no variation \n"%(arr[0]+": "+arr[1]))
						filterp += 1
						continue## filter the no var data
					if np.sum(np.isnan(tmpdata)) > num:
						sys.stderr.write("[WARN] data: %s was filtered, too many NANs \n"%(arr[0]+": "+arr[1]))
						filterp += 1
						continue
					if np.sum(np.isnan(tmpdata)) + np.sum(tmpdata[~np.isnan(tmpdata)] < cutoff) > num:
						sys.stderr.write("[WARN] data: %s was filtered, too many exprs lower than noise \n"%(arr[0]+": "+arr[1]))
						filterp += 1
						continue
					if len(set(arr[2:])) <= 1:
						sys.stderr.write("[WARN] data: %s was filtered, because of no variation\n"%(arr[0]+": "+arr[1]))
						filterp += 1
						continue
				if log2tr:
					tmpdata = np.log2(tmpdata+addtolog)
					#tmpdata = np.log10(tmpdata+0.0000000000000000000000000000000001)
				realp += 1
				if realp % 100000 == 0:
					sys.stderr.write("[INFO] parsed %d data\n"%realp)
				self.data[:,realp-1] = tmpdata
				self.anno.append(arr[0] + "\t" + arr[1])
				self.annosep.append(arr[0] + "|" + arr[1])
				self.anno1.append(arr[0])
				self.anno2.append(arr[1])
		#self.data = np.asmatrix(np.transpose(self.data.reshape(self.p,self.n)))
		#filter the sd 
		sys.stderr.write("[INFO] filter numbers: %d\n"%filterp)
		sys.stderr.write("[INFO] real numbers: %d\n"%realp)
		fh.close()
		# 2723,  4195,  8263,  8744, 11416
		self.data = np.asmatrix(self.data[:,0:realp])
		self.p = realp
		assert len(self.anno) == self.p
		sys.stderr.write("\n")
		sys.stderr.write('[INFO] Data Built done! cost %.2fs\n'%(time.time()-t0))
		return 0

class FactorFrame(object):
	def __init__(self):
		self.fnm = []##factor name
		self.snm = []##must same as sample infos
		self.lvs = 0## number of variables
		self.var = []##
		self.levels = []
	def parse_factor(self,factorfile):
		f = compress.gz_file(factorfile,"r")
		for line in f:
			if line.startswith("##"):continue
			if line.startswith("#"):
				self.fnm = line.rstrip("\n").split("\t")[1:]
				self.lvs = len(self.fnm)
				self.levels = [0,]*self.lvs
				continue
			arr = line.rstrip("\n").split("\t")
			self.snm.append(arr[0])
			self.var.append(map(str,arr[1:]))
		f.close()
		self.var = np.asarray(self.var)
		for i in range(self.lvs):
			self.levels[i] = len(set(self.var[:,i].tolist()))
		print(self.levels)
		self.var = np.float64(self.var)
		return 0

def datacheck(X):
	ret = 1
	if isinstance(X,np.matrix) or isinstance(X,np.array):pass
	else:
		sys.stderr.write('[ERROR] centring only support matrix or array data\n')
		return ret
	if X.dtype == np.float64 or X.dtype == np.float32:pass
	else:
		sys.stderr.write('[ERROR] centring only support float64 or float32 data\n')
		return ret
	return 0

def centring(X,axis=0):
	ret = 1
	#if datacheck(X):
	#	return ret
	Xmean = np.mean(X,axis=axis)
	X -= Xmean
	return Xmean
def mad(X,axis=None):
	Xmad = np.median(np.abs(X-np.median(X,axis=axis)),axis=axis)
	return Xmad

def normalize(X,axis=0):
	ret = 1
	#if datacheck(X):
	#	return ret
	Xstd =np.std(X,ddof=1,axis=axis)
	#print np.where(Xstd==0)
	X /= Xstd
	return Xstd


def twoDimDistr(object, fraction = 100):
	''' 
	Object must be a matrix format, and the number of columns is 2.
	'''
	if not isinstance(object, np.ndarray):
		try:object = np.asarray(object)
		except:
			raise
		else:sys.stderr.write('[WARN] Transfer the raw data to matrix format.\n')
	if object.shape[-1] != 2: return 1
	#for i in range(np.shape(object)[-1]):
	#	if np.max(abs(object[:, i])) > 1:
	#		object[:, i] = object[:, i] / np.max(abs(object[:, i]))
	xmin = -1#np.min(object[:,0]);
	xmax = 1 #np.max(object[:,0]);
	ymin = -1#np.min(object[:,1]);
	ymax = 1 #np.max(object[:,1])
	x_coord = np.linspace(xmin, xmax, fraction).tolist()
	y_coord = np.linspace(ymin, ymax, fraction).tolist()
	x_coord.append(np.inf)
	y_coord.append(np.inf)
	countsMatrix = np.zeros((fraction, fraction))
	for i in range(fraction):
		bool_x = np.int32(object[:,0] >= x_coord[i]) * np.int32(object[:,0] < x_coord[i+1])
		#x_idx = np.bool8(bool_x)
		for j in range(fraction):
			#tmp_y = object[:,1][x_idx]
			#bool_x = np.int32(object[:,0] >= x_coord[i]) * np.int32(object[:,0] < x_coord[i+1])
			bool_y = np.int32(object[:,1] >= y_coord[j]) * np.int32(object[:,1] < y_coord[j+1])
			countsMatrix[i,j] = np.sum(bool_x * bool_y)
	return countsMatrix,[xmin,xmax,ymin,ymax]

def quantile(Xnp):
	ranks=[]
	n, p = Xnp.shape
	Xnormalize = np.asmatrix(np.zeros((n,p)))
	for i in range(n):
		ranks.append(np.int32(stats.rankdata(Xnp[i,:],"min")).tolist())
	Xnptmp = Xnp.copy()
	Xnptmp.sort()
	ranks = np.asarray(ranks)
	Xnptmpmean = np.asarray(np.mean(Xnptmp,axis = 0))
	for i in range(n):
		Xnormalize[i] = Xnptmpmean[0][ranks[i]-1]
	return Xnormalize

##int get_permutation(char *vipname,char *pername,long c_demensionx_adv,long c_ntimes,double *pvalue_adv)
def c_permutation(fvipname,fpername,p,ntimes=1000):
	pvalue = None
	try:
		prefix =  os.path.dirname(os.path.abspath(__file__))
		sys.stderr.write('[INFO] Clib load: "%s"\n'%prefix)
		libpremutation  = ctypes.CDLL(os.path.sep.join([prefix,"permutation",'cal_permutation.so']))
	except:
		sys.stderr.write('[ERROR] Can not load cal_permutation.so\n')
		return pvalue
	pvalue = (ctypes.c_double * p)()
	ctypes.memset(ctypes.addressof(pvalue),0,ctypes.sizeof(pvalue))
	ret = libpremutation.get_permutation(ctypes.c_char_p(fvipname),ctypes.c_char_p(fpername),ctypes.c_long(p),ctypes.c_long(ntimes),pvalue)
	return pvalue

def comb_replace(datalist,num):
	return list(itertools.combinations_with_replacement([datalist], num))

if __name__ == "__main__":
	##can use abspath to get __file__ path ,and then load module
	libpremutation  = ctypes.CDLL('cal_permutation.so')
	print(libpremutation.get_permutation)
	#a = np.array([[5,4,3],[2,1,4],[3,4,6],[4,2,8]])
	#a= np.asmatrix(np.transpose(a))
	#print quantile(a)
	ins = FactorFrame()
	ins.parse_factor(sys.argv[1])
