import sys
import os

from rblib import seqio,bamio,defs
from sklearn.decomposition import NMF
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from rblib import statplot
import pysam
from  Bio.Seq import Seq

class Signature(object):
	def __init__(self,strand=0):
		if strand:
			f = open(os.path.dirname(os.path.abspath(__file__))+os.path.sep+"ref.types.strands","r")
			self.signumber = 192
			self.mutlabel = ["A>C","A>G","A>T","C>A","C>G","C>T","G>A","G>C","G>T","T>A","T>C","T>G"]
		else:
			f = open(os.path.dirname(os.path.abspath(__file__))+os.path.sep+"ref.types","r")
			self.signumber = 96
			self.mutlabel = ["C>A/G>T","C>G/G>C","C>T/G>A","T>A/A>T","T>C/A>G","T>G/A>C"]

		self.context = []
		self.ref     = []
		self.alt     = []
		self.hsigreftypeidx    = {}

		self.genomeidx = None
		self.bed = []

		self.APOBEC = dict.fromkeys([28,44,31,47])# 
		for line in f:
			arr = line.rstrip("\n").split("\t")
			self.context.append(arr[0])
			self.ref.append(arr[1])
			self.alt.append(arr[2])
			self.hsigreftypeidx["%s:%s>%s"%(arr[0],arr[1],arr[2])] = int(arr[3])
		f.close()
		#print self.hsigreftypeidx,self.context,self.ref,self.alt
	def getgenome(self,genome):
		self.genomeidx = seqio.faidx(genome)
		return 0
	def getbed(self,bed):
		tmpbed_plus  = []
		tmpbed_minus = []
		hstrand = {"+":0,"-":1}
		for ret in seqio.bed6_parse(bed):
			chrom,intstart,intend,name,score,strandother  = ret
			[tmpbed_plus,tmpbed_minus][hstrand[strandother]].append([chrom,intstart,intend])
			#tmpbed.append([chrom,intstart,intend])

		mergedregion_plus,totallen_plus = seqio.merge_chromregion(tmpbed_plus)
		mergedregion_minus,totallen_minus = seqio.merge_chromregion(tmpbed_minus)
		for i in mergedregion_plus:
			i.append("+")
		for i in mergedregion_minus:
			i.append("-")
		self.bed = mergedregion_plus + mergedregion_minus
		#print self.bed[0],self.bed[-1]

	def process_context_strand(self,chrom,pos,ref,alts,strand = "+"):
		mutmatrix_idxs = []
		keysret = []

		for alt in alts:
			if ref == alt:continue
			#try:
			triseq = self.genomeidx.fetch(chrom,pos-1,pos+2).upper()
			
			if strand == "-":
				try:
					ref = defs.hrc[ref]; alt = defs.hrc[alt]
				except:
					continue
				triseq = str(Seq(triseq).reverse_complement())
				#mutmatrix_idxs.append(-1) # -1 to record nothing
				#print triseq
			#	continue
			keytmp = "%s:%s>%s"%(triseq,ref,alt)
			keysret.append(keytmp)
			#print keytmp
			if keytmp in self.hsigreftypeidx:
				idx = self.hsigreftypeidx[keytmp]
				mutmatrix_idxs.append(idx)
				#print idx
			#else:
			#	mutmatrix_idxs.append(-1)
		return mutmatrix_idxs,keysret

	def process_context(self,chrom,pos,ref,alts,strand="."):# pos is 0-based 
		mutmatrix_idxs = []
		keysret = []
		for alt in alts:
			if ref == alt:continue
			triseq = self.genomeidx.fetch(chrom,pos-1,pos+2).upper()
			if ref == "C" or ref == "T":pass ## use a expand dict to avoid if condition # 
			else:
				try:
					ref = defs.hrc[ref]; alt = defs.hrc[alt]
				except:
					#mutmatrix_idxs.append(-1) # -1 to record nothing
					continue
				triseq = str(Seq(triseq).reverse_complement())
			keytmp = "%s:%s>%s"%(triseq,ref,alt)
			keysret.append(keytmp)
			if keytmp in self.hsigreftypeidx: ## need a additoonal to process not int dict
				idx = self.hsigreftypeidx[keytmp]
				mutmatrix_idxs.append(idx)
			#else:
				#mutmatrix_idxs.append(-1)
		return mutmatrix_idxs,keysret

	def getmutation_pattern_maf(self,fmaf): # do without the bed file, filter by user self
		mutmatrix = np.zeros(self.signumber)
		fmaf = open(sys.argv[1],"r")
		for line in fmaf:
			if line.startswith("#") or line.startswith("Hugo_Symbol"):continue
			arr = line.rstrip("\n").split("\t")
			snptype = arr[9]
			if snptype != "SNP":continue
			chrom = arr[4]
			pos = int(arr[5]) - 1
			ref = arr[10]
			alts = [arr[11],arr[12]]
			mutmatrix_idxs = self.process_context(chrom,pos,ref,alts,".")[0]### add [0]
			for idx in mutmatrix_idxs:
				mutmatrix[idx] += 1
		fmaf.close()
		return mutmatrix
	def process_APOBEC(self,mutmatrix_idxs): # 
		apobec_bool = []
		for mutmatrix_idx in mutmatrix_idxs:
			try:
				if mutmatrix_idx in self.APOBEC:
					apobec_bool.append(True)
				else:
					apobec_bool.append(False)
			except:
				apobec_bool.append(False)
		return apobec_bool

	def getmutation_pattern(self,fbvcf,flagstrand=0):# use only protein_coding genes exom region !!
		# first to screen the bvcf and to get variants in beds 
		try:
			tabix = bamio.tabixIO(fbvcf,seq_col=0,start_col=1,end_col=1)
		except:
			tabix = bamio.tabixIO(fbvcf+".gz",seq_col=0,start_col=1,end_col=1)
		contigs =  dict.fromkeys(tabix.contigs)
		mutmatrix = np.zeros(self.signumber)

		### 
		process_context_func = [self.process_context,self.process_context_strand][flagstrand]

		for chrom,intstart,intend,strand in self.bed:
			if chrom not in contigs:continue
			it = tabix.fetch(chrom,intstart,intend)
			for recline in it:
				#varrecs = bvcf.fetch(chrom,intstart,intend)	
				arr = recline.rstrip("\n").split("\t")
				chrom = arr[0]
				pos = arr[1]
				ref = arr[3].upper()
				alts = arr[4].upper().split(",")
				pos = int(pos)-1 # pos is 1-based inclusive
				mutmatrix_idxs = process_context_func(chrom,pos,ref,alts,strand)[0]
				for idx in mutmatrix_idxs:
					#print idx
					mutmatrix[idx] += 1
			del it
		tabix.close()
		return mutmatrix

	def doNMF(self,X,n_components=None, init=None, solver='cd', tol=0.0001, max_iter=200, random_state=None, alpha=0.0, l1_ratio=0.0, verbose=0, shuffle=False, nls_max_iter=2000, sparseness=None, beta=1, eta=0.1):
		nmf = NMF(n_components=n_components, init=init, solver=solver, tol=tol, max_iter=max_iter, random_state= random_state, alpha=alpha, l1_ratio=l1_ratio, verbose=verbose,shuffle=shuffle, nls_max_iter=nls_max_iter, sparseness=sparseness, beta=beta, eta=eta)
		sdist = nmf.fit_transform(X)#[nsamles, n_components]
		vdist = nmf.components_ ##  [n_components, n_features]
		reconstruction_err = nmf.reconstruction_err_

		return sdist,vdist,reconstruction_err
		
	
	def plotreconstruction_err(self,X,figprefix="NMF_Reconstruction_Error",minnum=1,maxnum=10,init=None, solver='cd', tol=0.0001, max_iter=200, random_state=None, alpha=0.0, l1_ratio=0.0, verbose=0, shuffle=False, nls_max_iter=2000, sparseness=None, beta=1, eta=0.1):
		ret = np.zeros(maxnum-minnum+1)
		for i in range(minnum,maxnum+1):
			nmf = NMF(n_components=i)
			sdist = nmf.fit_transform(X)
			vdist = nmf.components_
			reconstruction_err = nmf.reconstruction_err_
			ret[i-1]  = reconstruction_err
		#toterr = np.sqrt(np.sum((X-0)**2))
		statplot.plotline(np.arange(minnum,maxnum+1),np.asarray([ret.tolist(),]),figprefix,"Number of signatures","Reconstruction Error",['bo--'],xlimmax=maxnum+1,)
# 1. get genome faidx ,  get context 
# 2. do NMF ? mean center or normlize 
# 3. plot signature both samples and Signatures
# 4. raw variant on genes for C>G ... six  

	def __clean_x(self,ax):
		ax.get_xaxis().set_ticks([])
	def __clean_y(self,ax):
		ax.get_yaxis().set_ticks([])

	def __clean_axis(self,ax):
		"""Remove ticks, tick labels, and frame from axis"""
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
		#for spx in ax.spines.values():
		#	spx.set_visible(False)
	
	def norm(self,data):
		n,p = data.shape
		datasum = np.sum(data,axis=1)
		datanorm = np.zeros((n,p))
		for i in range(n):
			for j in range(len(self.mutlabel)): ## 
				datanorm[i,16*j:16*(j+1)]= data[i,16*j:16*(j+1)]/datasum[i]
		datanorm[np.isnan(datanorm)] = 0.0
		return datanorm


	def plotMutSpectrum(self,data,gnames,width=0.8,fig_prefix="mutatation_profile96"):
		# data:  nsn or ngn * nfeatures 
		n,p = data.shape
		ret_colors,ret_lines,ret_markers = statplot.styles(len(self.mutlabel))
		maxdat = np.max(data)
		tmpheightsize =  n*1.2
		if tmpheightsize < 4:
			tmpheightsize = 4

		fig = plt.figure(figsize=(p/6,tmpheightsize),dpi=300)
		datasum = np.sum(data,axis=1)
		maxdat = maxdat/np.max(datasum)	
		for i in range(n):
			for j in range(len(self.mutlabel)): #
				ax = fig.add_subplot(n,len(self.mutlabel),i*len(self.mutlabel)+j+1)
				xlocations = np.arange(16)
				ax.bar(xlocations, data[i,16*j:16*(j+1)]/datasum[i],width=width,linewidth=0,color=ret_colors[j],ecolor=ret_colors[j],alpha=0.6)
				ax.plot((0,16),(1.0/self.signumber,1.0/self.signumber),'r--')
				ax.set_ylim(ymax=maxdat)
				if i == 0:ax.set_title(self.mutlabel[j]) ### "%.2f%%"%np.sum(data[i,16*j:16*(j+1)])/datasum[i])
				if j != 0:
					self.__clean_y(ax)
				else:
					ax.set_ylabel(gnames[i])
					tickL = ax.yaxis.get_ticklabels()
					for t in tickL:
						t.set_fontsize(t.get_fontsize() - 3)
				if i == n-1: 
					ax.set_xticks(xlocations+width/2)## 0.2 is width 
					ax.set_xticklabels(self.context[16*j:16*(j+1)],rotation=90)
				else:
					self.__clean_x(ax)

		fig.tight_layout()
		plt.savefig(fig_prefix+".png",format='png',dpi=300)
		plt.savefig(fig_prefix+".svg",format='svg',dpi=300)
		plt.clf()
		plt.close()
		return 0
	

def sampleinfo(fn):
	#sn	group	filenamepath	
	f = open(fn,"r")
	hsn = {}
	hgn = {}
	snarr = []
	gnarr = []
	agnarr = []	
	for line in f:
		arr = line.rstrip("\n").split("\t")
		if arr[0] not in hsn:
			snarr.append(arr[0])
			hsn[arr[0]] = []
			agnarr.append(arr[1])
		hsn[arr[0]].append(arr[2])
		if arr[1] not in hgn:
			gnarr.append(arr[1])
			hgn[arr[1]] = []
		hgn[arr[1]].append(arr[2])
	f.close()
	return hsn,hgn,snarr,gnarr,agnarr
if __name__ == "__main__":
	signature = Signature()
	

	# test for NMF 
	#RATE_MATRIX = np.array([[5, 5, 3, 0, 5, 5],[5, 0, 4, 0, 4, 4],[0, 3, 0, 5, 4, 5],[5, 4, 3, 3, 5, 5]])
	#sdist,vdist,reconstruction_err =  signature.doNMF(RATE_MATRIX,n_components=10)
	#np.dot(RATE_MATRIX * vdist[])
	#signature.plotreconstruction_err(RATE_MATRIX,minnum=1,maxnum=4)	
	#print sdist
	#print vdist
	#print reconstruction_err
	
	
	### test for genome mutation profile
	signature.getgenome(sys.argv[1])
	#signature.getbed(sys.argv[2])
	hsn,hgn,snarr,gnarr =  sampleinfo(sys.argv[3]) 
	usesn    = 1
	usegroup = 1

	##### do sum mutation rate 
	try:
		tmpdat = np.load("signature.saved.dat.npz")
	except IOError,e:
		sys.stderr.write("[WARN] %s\n"%str(e).strip())
		signature.getbed(sys.argv[2])
		if usesn:
			sntotmutmatrix = np.zeros((len(snarr),96))
			for i in range(len(snarr)):
				sn = snarr[i]
				tmpfiles = hsn[sn]
				print(tmpfiles)
				kmutmatrix = np.zeros(96)
				for tmpfile in tmpfiles:
					mutmatrix = signature.getmutation_pattern(tmpfile)
					kmutmatrix += mutmatrix
				#print kmutmatrix
				sys.stderr.write("[INFO] parsed %s\n"%sn)
				sntotmutmatrix[i,:] = kmutmatrix[:]
		if usegroup:
			gntotmutmatrix = np.zeros((len(gnarr),96))
			for i in range(len(gnarr)):
				gn = gnarr[i]
				tmpfiles = hgn[gn]
				print(tmpfiles)
				kmutmatrix = np.zeros(96)
				for tmpfile in tmpfiles:
					mutmatrix = signature.getmutation_pattern(tmpfile)
					kmutmatrix += mutmatrix
				gntotmutmatrix[i,:] = kmutmatrix[:]
		np.savez("signature.saved.dat",sntotmutmatrix=sntotmutmatrix,gntotmutmatrix=gntotmutmatrix)
		tmpdat = np.load("signature.saved.dat.npz")
	
	## 
	# to plot for each groups !, raw fractions of each groups
	# to get a numgn * 96 matrix to plot # 96 groups will in different 
	
	# plot group MutSpectrum , 
	try:
		signature.plotMutSpectrum(tmpdat["gntotmutmatrix"],gnarr)
	except:pass

	sndatanorm = signature.norm(tmpdat["sntotmutmatrix"])
	
	# plot snample Reconstruction_Error
	try:
		signature.plotreconstruction_err(sndatanorm,"samples_NMF_Reconstruction_Error",maxnum=20)
	except:pass

	# plot group Reconstruction_Error
	gndatanorm = signature.norm(tmpdat["gntotmutmatrix"])
	try:
		signature.plotreconstruction_err(sndatanorm,"groups_NMF_Reconstruction_Error",maxnum=6)
	except:
		pass
	
	## use last ncmp to do NMF
	lastncom = 6
	nmf = NMF(n_components=lastncom)
	#print sndatanorm
	sdist = nmf.fit_transform(sndatanorm)
	vdist = nmf.components_
	signame = []
	for i in range(lastncom):signame.append("signature%d"%(i+1))
	## plot mutation spectrum 
	signature.plotMutSpectrum(vdist,signame,fig_prefix="signature_MutSpectrum")
	## plot samples signaure composition
	statplot.cluster_stackv_bar_plot(np.asmatrix(sdist).T,snarr,"samples_signature_composition","xlabel","ylabel",width=0.9,legends=signame,scale=0,rotation=90,nocluster=0,noline=1)
