# -*- coding: UTF-8 -*-
import sys
#use the stem like to estimate 
# Clustering short time series gene expression data
# Ernst, J.; Nau, G. J.; Bar-Joseph, Z., Clustering short time series gene expression data. Bioinformatics 2005, 21 Suppl 1, i159-68.
import numpy as np
import itertools
from scipy import misc
from mutilstats import comb_replace
from enrich import fdr_core
import statplot
from rklib.progress_bar import ProgressBar
def exprs_profile(ntimes,c=1):
	# c is default = 1
	# (2c + 1) ** (n-1)
	starts = 0
	if ntimes < 2:
		sys.stderr.write("[ERROR] Ntimes must be >= 2, please check\n")
		sys.exit(1)
	c_range = list(range(-1*c,c+1))
	prod_arr = [[0],]
	for i in range(ntimes-1):
		prod_arr.append(c_range)
	profile_matrix = []
	for i in itertools.product(*prod_arr):
		profile_matrix.append(i)
	profile_matrix = np.cumsum(np.asarray(profile_matrix),axis=1)
	m,ntimes = profile_matrix.shape
	#print profile_matrix,  the min for start merge is matrix[0,:]
	#print profile_matrix
	if m > 50:
		profile_matrix = merge_profile(profile_matrix)
	m,ntimes = profile_matrix.shape
	sys.stderr.write("[INFO] select %d profiles\n"%m)
	profileslables = []
	zerolabel =  ",".join(['0']*ntimes)
	profile_matrix_out = []
	for i in range(m):
		tmplabel = ",".join(map(str,profile_matrix[i,:].tolist()))
		if tmplabel == zerolabel: continue
		profileslables.append(tmplabel)
		profile_matrix_out.append(profile_matrix[i,:].tolist())
	#profileslables = np.asarray(profileslables)
	profile_matrix_out.append([0,]*ntimes)
	profileslables.append(zerolabel)
	return np.asarray(profile_matrix_out),profileslables

from gwaspls import kenstone
def merge_profile(profiles,num=50):
	# matrix,  m  , and d cut
	# use the corr 
	# 用一列，True and False to 划分 2个不同的中选组， 用逻辑算符  直接从大矩阵中选择小矩阵
	model,other = kenstone(profiles,[],num)
	return profiles[model,:]

def assign_profile(exprs_matrix,profiles,profileslables):
	#profileslables = np.asarray(profileslables)
	exprs_matrix = np.asarray(exprs_matrix)
	ngenes,p1 = exprs_matrix.shape
	m,p2 = profiles.shape
	assert p1 == p2
	class_observe_counts = np.zeros(m)# use weight by more than 2 similar class
	trend_idx = {}
	midx = np.asarray(list(range(m-1)))# now change "range(m)" to "range(m-1)"
	for i in range(ngenes):
		if np.sum(exprs_matrix[i,:] == 0.0) >=p1:
			trend_idx[i] = [m-1,]
			class_observe_counts[-1] += 1
			continue
		coefmat = np.corrcoef(exprs_matrix[i,:],profiles[0:-1,:])
		idx = coefmat[0,1:] == np.nanmax(coefmat[0,1:])
		class_observe_counts[0:-1] += np.float64(idx)/np.sum(idx)
		#trend_idx[i] =
		#tmp_trend_idx = ";".join(profileslables[idx].tolist())
		tmp_trend_idx = midx[idx].tolist()
		trend_idx[i] = tmp_trend_idx
	# do the test
	return trend_idx,class_observe_counts
from scipy import stats
def permute_bernoulli(exprs_matrix,profiles,raw_class_observe_counts):
	ngenes,p1 = exprs_matrix.shape
	## first to shuttle the array
	m,p2 = profiles.shape
	class_observe_counts = np.zeros(m)
	idx = list(range(p1))
	np.random.shuffle(idx)
	exprs_matrix_0 = exprs_matrix[:,idx]
	for i in itertools.permutations(list(range(p1)), p1):
		idx = np.asarray(i)
		exprs_matrix_1 = exprs_matrix_0[:,idx]
		for i in range(ngenes):
			if np.sum(exprs_matrix_1[i,:] == 0.0) >=p1:
				class_observe_counts[-1] += 1
				continue
			coefmat = np.corrcoef(exprs_matrix_1[i,:],profiles[0:-1,:])
			idx = coefmat[0,1:] == np.nanmax(coefmat[0,1:])
			#print idx
			class_observe_counts[0:-1] += np.float64(idx)/np.sum(idx)
	class_observe_counts = class_observe_counts/misc.factorial(p1)
	retvalue = []
	for i in range(m):
		tmp_p = class_observe_counts[i]/ngenes
		tmp_counts = raw_class_observe_counts[i]
		# get the right tail 
		rawp = 1 - stats.binom.cdf(tmp_counts, ngenes, tmp_p)
		#print tmp_counts,ngenes,tmp_p
		#print rawp
		retvalue.append([i,rawp,0,tmp_counts])
	# cal qvalue
	retvalue = fdr_core(retvalue,2,1)
	retdict = {}
	for tmp in retvalue:
		retdict[tmp[0]] = tmp[1:]
	return retdict

def short_time_analysis(meandata_arr,c=1):
	meandata = np.asarray(meandata_arr)
	ngenes,ntimes = meandata.shape
	profile_matrix,profileslables = exprs_profile(ntimes,c=c)
	trend_idx,class_observe_counts = assign_profile(meandata,profile_matrix,profileslables) 
	retdict = permute_bernoulli(meandata,profile_matrix,class_observe_counts)
	# output
	retdata = {}
	for i in range(ngenes):##  out is  geneid --> profilelabels:pvalue:qvalue:counts
		genetrendidx = trend_idx[i]
		#print genetrendidx
		strout = []
		for tmpgenetrendidx in genetrendidx:
			tmplabel = profileslables[tmpgenetrendidx]
			rawp,qvalue,rawcounts = retdict[tmpgenetrendidx]	
			strout.append(tmplabel+":"+str(rawp)+":"+str(qvalue)+":"+str(rawcounts))### the rawcounts is a estimator
		retdata[i] = ";".join(strout)
	return retdata

def parse_edgeRdir(num,edgeRdir,prefix="Root.matrix.xls",subpre="cond_",starts=1):
	h={}
	for i in range(starts,num+starts):
		h[i-starts] = {}
		for j in range(i+1,num+starts):
			print(i,j)
			h[i-starts][j-starts] = {}
			fn = edgeRdir+ "/"+ prefix + "." + subpre+str(i)+"_vs_"+subpre+str(j)+".edgeR.DE_results"
			f = open(fn,"r")
			f.next()
			for line in f:
				name,logfc,logCPM,PValue,fdr = line.strip().split("\t")
				name = name.split("|")[-1]
				h[i-starts][j-starts][name] = float(PValue)
			f.close()
	return h

import mutilstats
from rklib.fmtprint import f2str
from rklib import utils

def get_compare_fornorepeat(matrix,sampleinfo,edgeRdir,outdir="./",prefix="Root.matrix.xls",subpre="cond_",starts=1,plotflag=0):
	try:
		utils.dirDetectCreate(outdir)
		utils.dirDetectCreate(outdir+"/DiffPlot")
	except:return 1
	sinfo = mutilstats.SampleInfo()
	sinfo.parse_sampleinfo(sampleinfo)
	num = len(sinfo.sns)
	data = mutilstats.MatrixAnno()
	data.parse_matrix_anno(matrix,addtolog=1,log2tr=1)
	hsig = parse_edgeRdir(num,edgeRdir,prefix=prefix,subpre = subpre,starts=starts)
	tot_mean = []
	letterlist = list("abcdefghijklmnopqrstuvwxyz")
	result_tot = []
	pb = ProgressBar(xmax=data.p)
	for pi in range(data.p):
		pb()
		meanarr = np.asarray(data.data[:,pi]).T[0].tolist()
		tot_mean.append(meanarr)
		meansort = []
		for idx in range(len(meanarr)):
			meansort.append([meanarr[idx],1,idx,1,set([])])
		mean_fc2ck_strlist = []
		for k in range(len(meanarr)):
			mean_fc2ck_strlist.append(f2str(meanarr[k]-meanarr[0]))
	## mut com 
		meansorted = utils.us_sort(meansort,0)
		meansorted[0][-1].add('a')
		letteridx = 0
		smeanidx  = 0
		for i in range(1,len(meansorted)):
			ii = meansorted[smeanidx][2]
			jj = meansorted[i][2]
			iii = min(ii,jj)
			jjj = max(ii,jj)
			assert iii != jjj
			if data.anno2[pi] in hsig[iii][jjj]:
				prob = hsig[iii][jjj][data.anno2[pi]]
			else:
				prob = 1.0
			if prob >= 0.05:
				meansorted[i][-1].add(letterlist[letteridx])
				#print letterlist[letteridx]
				continue
			else:
				letteridx += 1
				marker = letterlist[letteridx]
				#print marker
				meansorted[i][-1].add(marker)
				for j in range(i-1,smeanidx,-1):
					ii =  meansorted[j][2]
					iii = min(ii,jj)
					jjj = max(ii,jj)
					assert iii != jjj
					if data.anno2[pi] in hsig[iii][jjj]:
						prob = hsig[iii][jjj][data.anno2[pi]]
					else:
						prob = 1.0
					if prob >= 0.05:
						meansorted[j][-1].add(marker)
						smeanidx = j
					else:continue
		meansort = utils.us_sort(meansorted,2)
		markers = []
		for a in meansort:
			markers.append("".join(sorted(list(a[-1]))))
		# meansort,",".join(markers)
		marklabel = ",".join(markers)
		Fstat,pvalue = ["-","-"]
		stdarr = np.zeros(num)
		ylabel = "$Normalized\ Expression(log_{2}\ transformed)$"
		if plotflag:
			statplot.onefactor_diff_plot(meanarr,stdarr,sinfo.uniqclasslabel,fig_prefix=outdir+"/DiffPlot/"+data.anno1[pi]+"_"+data.anno2[pi],title=data.anno1[pi]+" : "+data.anno2[pi]+":"+marklabel,ylabel=ylabel)
		else:pass
		meanstdarr = []
		for ii in range(len(meanarr)):
			meanstdarr.append("%.3f"%meanarr[ii]+"+-"+"%.3f"%stdarr[ii])
		result_tot.append([data.anno1[pi],data.anno2[pi],Fstat,pvalue,0,np.mean(meanarr),pi,marklabel]+meanstdarr+mean_fc2ck_strlist)
	pb.end()
	# do time
	retdata = short_time_analysis(np.asarray(tot_mean),c=1)
	for i in range(len(result_tot)):
		geneid = result_tot[i][6]
		siglabel = retdata[geneid]
		result_tot[i][6] = siglabel
	#result_tot_sort = fdr_core(result_tot,4,3)
	#del result_tot
	## out put 
	h_data = {}
	floatfmt = "\t".join(["%s"]*data.n)
	for i in range(data.p):
		key = data.anno1[i]+"\t"+data.anno2[i]
		tmpdata = tuple(np.asarray(data.data[:,i]).T[0].tolist())
		h_data[key] = floatfmt%tmpdata
	mean_arr_header = []
	fc_arr_header = []
	for classlabel in sinfo.uniqclasslabel:
		fc_arr_header.append("log2FC(%s/%s)"%(classlabel,sinfo.uniqclasslabel[0]))
		mean_arr_header.append(classlabel+"(mean)")
	fhout_sig = open(outdir+"/"+"no_repeat_compare"+".sig.xls","w")
	fhout_sig_mat = open(outdir+"/"+"no_repeat_compare"+".sig.anno","w")
	fhout_sig.write("##significant difference list\n")
	fhout_sig.write("#ID\tmatureID\tF statistics\tpvalue\tqvalue\tExprsMean\tTrend_label\tMultiple_comparison"+"\t"+"\t".join(mean_arr_header)+"\t"+"\t".join(fc_arr_header)+"\n")
	for i in range(len(result_tot)):
		nameid,anno,statics,pvalue,fdr = result_tot[i][0:5]
		strout = "\t".join(map(str,result_tot[i]))
		fhout_sig.write(strout+"\n")
		tmpdata = h_data[nameid+"\t"+anno]
		fhout_sig_mat.write(nameid+"\t"+anno+"\t"+tmpdata+"\n")
	fhout_sig.close()
	fhout_sig_mat.close()
	return 0

if __name__ == "__main__":
	#print exprs_profile(4,2)
	"""
	randomnum = np.zeros((500,4))
	randomnum[:,1:] += (np.random.random((500,3))-0.5)*3
	retdata = short_time_analysis(randomnum)
	#print randomnum
	for i in retdata:
		print i,retdata[i]
	"""
	#get_compare_fornorepeat(matrix,sampleinfo,edgeRdir,outdir="./",prefix="Root.matrix.xls",subpre="cond_",starts=1)
	# def get_compare_fornorepeat(matrix,sampleinfo,edgeRdir,outdir="./",prefix="Root.matrix.xls",subpre="cond_",starts=1,plotflag=0):  
	get_compare_fornorepeat(sys.argv[1],sys.argv[2],sys.argv[3],outdir="./",prefix=sys.argv[4],subpre=sys.argv[6],starts=0,plotflag=int(sys.argv[5]))
# a = [1,3,4,5,6]
# np.random.shuffle(a) 
# a is [4, 1, 6, 3, 5]  
