import sys
from rklib.utils import us_sort
from scipy import stats
import numpy as np
import pdb

def cal_pi0(p,lambda0 = None):
	##q-value in statistics, the minimum false discovery rate at which the test may be called significant
	# here we use the bootstrap method to estimate pi0
	# for the list lambda0, otherwise, we use the lambda0 as 0.5
	p = np.asarray(p)
	m = len(p)
	err_str = "[INFO] The pi0 estimate for q-value is: %.3f\n"
	#first we use the list 0 : 0.05 : 0.90
	if lambda0 == None:
		lambda0 = np.arange(0,1.0,0.05)
		pi0 = np.zeros(len(lambda0))
		for i in range(len(lambda0)):
			pi0[i] = np.mean(p >= lambda0[i])/(1-lambda0[i])
		minpi0 = np.min(pi0)
		mse = np.zeros(len(lambda0))
		pi0_bootstrap = np.zeros(len(lambda0))
		for i in range(1000):
			try:
				p_bootstrap = np.random.choice(p,len(p),replace=True)
			except:
				return 1.0
			for i in range(len(lambda0)):
				pi0_bootstrap[i] = np.mean(p_bootstrap>lambda0[i])/(1-lambda0[i])
			mse += (pi0_bootstrap-minpi0)**2
		pi0 = np.min(pi0[mse == np.min(mse)])
		pi0 = min(pi0,1.0)
		if pi0 <= 0:
			sys.stderr.write("[WARN] bootstrap estimate failed, Check that you have valid p-values, PROG will try another method\n")
		else:
			sys.stderr.write(err_str%pi0)
			return pi0
	lambda0 = 0.5##  0 < lambda0 < 1
	#if not isinstance(lambda0,list):
	#	assert isinstance(lambda0,float)
	pi0 = np.mean(p >= lambda0)/(1 - lambda0)
	pi0 = min(pi0,1.0)
	if pi0 <= 0:
		sys.stderr.write("[WARN] PROG try to fail, Check that you have valid p-values, PROG will use Benjamini-Hochberg FDR\n")
		return 1.0
	else:
		sys.stderr.write(err_str%pi0)
		return pi0

def cummin_small_large(fdr_arr):
	#if fdr_arr[-1] >1.0:
	fdr_arr[-1] = min(1.0,fdr_arr[-1])
	arr_len = len(fdr_arr) -1
	for i in range(arr_len,0,-1):
		#if fdr_arr[i-1] > fdr_arr[i]:
		fdr_arr[i-1] = min(fdr_arr[i],fdr_arr[i-1])
	return fdr_arr

def fdr_core(result_tot,pos_fdr,pos_rawp,method="qvalue",robust=0):
	"""
	result_tot = [["gene1",10,0.05,0],["gene2",11,0.07,0]]  
	3
	2 
	return  result_totwithfdr
	"""
	if method not in ["fdr","qvalue"]:
		sys.stderr.write("[ERROR] Unkown multiple hypothesis testing method\n")
		return None
	if method == "fdr":pi0 = 1.0
	else:
		rawp = []
		for result in result_tot:
			rawp.append(result[pos_rawp])
		pi0 = cal_pi0(rawp)
	result_tot_sort = us_sort(result_tot,pos_rawp)
	result_tot_len = len(result_tot_sort)
	FDR_arr = []
	print("pi0 ret is",pi0)
	for i in range(result_tot_len):
		index = i+1
		if robust:
			FDR = result_tot_sort[i][pos_rawp]*pi0*result_tot_len/(index*(1-(1-result_tot_sort[i][pos_rawp])**result_tot_len))
		else:
			FDR = result_tot_sort[i][pos_rawp]*pi0*result_tot_len/index
		#print result_tot_sort[i][pos_rawp]
		FDR_arr.append(FDR)
		#print result_tot_sort[i][pos_rawp],FDR
	FDR_arr = cummin_small_large(FDR_arr)
	for i in range(result_tot_len):
		result_tot_sort[i][pos_fdr] = FDR_arr[i]
	return result_tot_sort

def do_enrich_core(siglist,totlist,hdb,method="qvalue",hstat={}):
	"""
	siglist  diff genes list [gene1,gene2,gene3]
	totlist                  [gene1,gene2,gene3,gen4,gene5,...]
	hdb[gene1] = ["GO:111","GO:222"]
	"""
	
	#if method not in ["fdr","qvalue"]:
	#	sys.stderr.write("[ERROR] Unkown multiple hypothesis testing method\n")
	#	exit(1)
	#if method == "fdr":
	#	pi0 = 1.0
	#else:
	#	pi0 = cal_pi0()
	set_siglist = set(siglist)
	set_totlist = set(totlist)
	siglist = list(set_siglist)
	totlist = list(set_totlist)
	henrich = {}
	hitermstat = {}
	if not hstat:
		for gene in hdb:
			iterms = hdb[gene]
			iterms = set(iterms)
			for iterm in iterms:
				if iterm in hitermstat: hitermstat[iterm] += 1
				else: hitermstat[iterm] = 1
	else:
		hitermstat = hstat.copy()
	M = 0;N=0;
	for siggene in siglist:
		if siggene in hdb:
			M += 1
			iterms = hdb[siggene]
			iterms = set(iterms)
			for iterm in iterms:
				if iterm in henrich:
					henrich[iterm][0].append(siggene)
				else:
					henrich[iterm] = [[],[],hitermstat[iterm]]
					henrich[iterm][0].append(siggene)
	back_list = set_totlist - set_siglist
	for totgene in back_list:
		if totgene in hdb:
			N += 1
			iterms = hdb[totgene]
			iterms = set(iterms)
			for iterm in iterms:
				if iterm in henrich:
					henrich[iterm][1].append(totgene)
				else:
					henrich[iterm] = [[],[],hitermstat[iterm]]
					henrich[iterm][1].append(totgene)
	result = []
	for iterm in henrich:
		m = len(henrich[iterm][0])
		#if m ==0:continue
		n = len(henrich[iterm][1])
		try:
			oddsratio, pvalue = stats.fisher_exact([[m,M-m],[n,N-n]],alternative="greater")
			if np.isinf(oddsratio):
				oddsratio, pvalue = stats.fisher_exact([[m,M-m],[n+1,N+1-n]],alternative="greater")
		except:
			sys.stderr.write("[ERROR] Scipy version too old\n")
			sys.exit(1)
		#result.append([iterm,oddsratio,pvalue,0.0,m,M,M-m,n,N,N-n])
		if m <=0:continue
		result.append([iterm,oddsratio,pvalue,0.0,m])
	#result = us_sort(result,2)
	#fdr_arr = []
	#result_len = len(result)
	#for i in range(result_len):
	#	idx = i+1
	#	fdr = result[i][2]*result_len/idx
	#	fdr_arr.append(fdr)
	#if fdr_arr:
	#	cummin_small_large(fdr_arr)
	#	for i in range(result_len):
	#		result[i][3] = fdr_arr[i]
	result_sort = fdr_core(result,3,2)
	#print henrich
	return result_sort,henrich

from rblib import statplot
def enrich_plot(result,pcut,nameidx,oddsratioidx,pvalueidx,namesubidx=None,namesubsep=None,fig_prefix="XXX.enrich",xlabel="$Odds Ratio$",ylabel="-$log_{10}(probability)$"):
	result_mark = []
	result_others = []
	flag = 1
	for dat in result:
		if namesubidx and namesubsep:
			name = dat[nameidx].split(namesubsep)[namesubidx]
		else:
			name = dat[nameidx]
		try:
			oddsratio = float(dat[oddsratioidx])
			pvalue = float(dat[pvalueidx])
		except:
			sys.stderr.write("[ERROR] can not parse")
			sys.stderr.write(str(dat))
			sys.exit(1)
		#print pvalue
		if pvalue <= pcut and flag ==1:
			result_mark.append([name,oddsratio,-1*np.log10(pvalue)])
		else:
			result_others.append([oddsratio,-1*np.log10(pvalue)])
		if len(result_mark) > 30:flag = 0
	statplot.plot_enrich(result_mark,result_others,fig_prefix,xlabel,ylabel)
	return 0

if __name__ == "__main__":
	#data = [[np.nan,0.001,0],]
	#data.append([2,0.013,0])
	#data.append([5,0.977,0])
	#data.append([np.inf,0.473,0])
	#data.append([4,0.872,0])
	#f = open(sys.argv[1],"r")
	
	#data = []
	#idx = 0
	#for line in f:
	#	p = float(line.strip())
	#	idx += 1
	#	data.append([idx,p,0])
	#result = fdr_core(data,2,1,method="qvalue",robust=0)
	#for i in result:
	#	print i
	result = open(sys.argv[1],"r")
	nameidx = int(sys.argv[2])
	fcidx  = int(sys.argv[3])
	pvalueidx = int(sys.argv[4])
	ret = []
	for line in result:
		if line.startswith("#"):continue
		arr = line.rstrip("\n").strip().split("\t")
		#arr[4] = -1*np.log10(float(arr[4]))
		ret.append(arr)
	enrich_plot(ret,0.05,nameidx,fcidx,pvalueidx,fig_prefix=sys.argv[1])
	#enrich_plot(ret,0.01,1,4,3,fig_prefix=sys.argv[1])
	# result,pcut,nameidx,oddsratioidx,pvalueidx,namesubidx=None,namesubsep=None
