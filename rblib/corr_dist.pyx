#-*- coding: UTF-8 -*-
import numpy as np
import scipy as sp
from scipy import stats
import sys
import itertools
import time
from rklib import mp
import pandas as pd
import h5py
def simplecall(func,arr1,arr2,i,j):
	idx = ~(np.isnan(arr1) | np.isnan(arr2))
	if np.sum(idx) >= 8:
		correlation,pvalue = func(arr1[idx],arr2[idx])
	else:
		#correlation = np.nan; pvalue = np.nan
		correlation = 0; pvalue = 1.0
	return correlation,pvalue,i,j

def process_result(results,p):
	result_corr = np.eye(p)
	result_pval = np.zeros((p,p))
	for correlation,pvalue,i,j in results:
		result_corr[i,j] = correlation; result_corr[j,i] = correlation;
		result_pval[i,j] = pvalue; result_pval[j,i] = pvalue
	return result_corr,result_pval

def simplecall2(arr1,dataset,k,method,tmppath):
	corr = stats.spearmanr if method else stats.pearsonr
	n,p = dataset.shape
	correlations = []
	pvalues      = []
	for i in xrange(p):
		tmpdata = dataset[:,i]
		idx = ~(np.isnan(arr1) | np.isnan(tmpdata))
		if np.sum(idx) >= 8:
			correlation,pvalue = corr(arr1[idx],tmpdata[idx])
		else:
			correlation = 0; pvalue = 1.0
		correlations.append(correlation)
		pvalues.append(pvalue)
	"""
	data = pd.DataFrame({
			"pvalues":np.asarray(pvalues),
			"correlations":np.asarray(correlations)
			})
	print data
	data.to_hdf('%s/tmpdata_%d.hdf'%(tmppath,k),'data')
	"""
	datafile = h5py.File('%s/tmpdata_%d.hdf'%(tmppath,k),'w')
	datafile.create_dataset('correlations', data = np.asarray(correlations))
	datafile.create_dataset('pvalues', data = np.asarray(pvalues))
	datafile.close()
	return 0

def nancorrmatrix_mp2(data1,data2,tmppath,method=0,nprocessor=8):
	sys.stderr.write("[INFO] use %s method\n"%["pearson","spearmanr"][bool(method)])
	n1,p1 = data1.shape
	n2,p2 = data2.shape
	assert n1 == n2
	process = mp.MPone(nprocessor)
	for i in xrange(p1):
		process.run(simplecall2,[data1[:,i],data2,i,method,tmppath])
	process.join()
	return 0

def nancorrmatrix_mp2_process(data1,data2,tmppath):
	n1,p1 = data1.shape
	n2,p2 = data2.shape
	assert n1 == n2
	Xcorr = np.zeros((p1,p2))
	pvalues = np.zeros((p1,p2))
	for i in xrange(p1):
		dataset = h5py.File('%s/tmpdata_%d.hdf'%(tmppath,i),'r')
		Xcorr[i,:]= dataset["correlations"][:]
		pvalues[i,:] = dataset["pvalues"][:]
		dataset.close()
	return Xcorr,pvalues

def nancorrmatrix_self_mp2(data,tmppath,method=0,nprocessor=8):
	sys.stderr.write("[INFO] use %s method\n"%["pearson","spearmanr"][bool(method)])
	n,p = data.shape
	process = mp.MPone(nprocessor)
	for i in xrange(p-1):
		process.run(simplecall2,[data[:,i],data[:,i+1:p],i,method,tmppath])
	process.join()
	return 0

def nancorrmatrix_self_mp2_process(data,tmppath):
	n,p = data.shape
	Xcorr = np.eye(p)
	pvalues = np.zeros((p,p))
	for i in xrange(p-1):
		dataset = h5py.File('%s/tmpdata_%d.hdf'%(tmppath,i),'r')
		Xcorr[i,i+1:] = dataset["correlations"][:]
		Xcorr[i+1:,i] = dataset["correlations"][:]
		pvalues[i,i+1:] = dataset["pvalues"][:]
		pvalues[i+1:,i] = dataset["pvalues"][:]
		dataset.close()
	return Xcorr,pvalues

def nancorrmatrix_self_mp(data,method=0,nprocessor=8):
	corr = stats.spearmanr if method else stats.pearsonr
	sys.stderr.write("[INFO] use %s method\n"%["pearson","spearmanr"][bool(method)])
	n,p = data.shape
	process = mp.MPone(nprocessor)
	for i in xrange(p-1): # 1 vs all ? 
		for j in xrange(i+1,p):
			process.run(simplecall,[corr,data[:,i],data[:,j],i,j])
	process.join()
	return process_result(process.results,p)
def nancorrmatrix_self(data,method=0):
	corr = stats.spearmanr if method else stats.pearsonr
	#print corr
	sys.stderr.write("[INFO] use %s method\n"%["pearson","spearmanr"][bool(method)])
	n,p = data.shape
	results = []
	for i in xrange(p-1):
		for j in xrange(i+1,p):
			#t0 = time.time()
			correlation,pvalue,i,j = simplecall(corr,data[:,i],data[:,j],i,j)
			#correlation,pvalue = corr(data[:,i],data[:,j])
			results.append([correlation,pvalue,i,j])
			#sys.stderr.write(str(time.time()-t0)+"\n")
	return process_result(results,p)


def nan_corr_matrix(data1,data2,method=0):
	# 计算变量间的相关性
	if method:
		corr = stats.spearmanr
		sys.stderr.write("[INFO] use the spearmanr corr method \n")
	else:
		corr = stats.pearsonr
		sys.stderr.write("[INFO] use the pearson corr method \n")
	n1,p1 = data1.shape
	n2,p2 = data2.shape
	assert n1 == n2
	corr_mat = np.asarray(np.zeros((p1,p2)))
	prob_mat = np.asarray(np.zeros((p1,p2)))
	for i in xrange(p1):
		sys.stderr.write("[INFO] process %d\n"%i)
		for j in xrange(p2):
			idx = np.isnan(data1[:,i]) | np.isnan(data2[:,j])	
			idx = ~np.asarray(idx.T)### [0]  use array 
			if np.sum(idx) > 6:
				correlation,pvalue = corr(data1[idx,i],data2[idx,j])
				#print correlation,pvalue
			else:
				correlation = np.nan
				pvalue = np.nan
			if np.isnan(correlation) or np.isnan(pvalue):
				correlation = 0.0
				pvalue = 1
			corr_mat[i,j] = correlation
			prob_mat[i,j] = pvalue
	return corr_mat,prob_mat

def corr_matrix(data1,data2,method=0):
	if method:
		corr = stats.spearmanr
		sys.stderr.write("[INFO] use the spearmanr corr method \n")
	else:
		corr = stats.pearsonr
		sys.stderr.write("[INFO] use the pearson corr method \n")
	n1,p1 = data1.shape
	n2,p2 = data2.shape
	assert n1 == n2 #samples is same  to call variants interaction
	corr_mat = np.asmatrix(np.zeros((p1,p2)))
	prob_mat = np.asmatrix(np.zeros((p1,p2)))
	for i in xrange(p1):
		sys.stderr.write("[INFO] process %d\n"%i)
		for j in xrange(p2):
			correlation,pvalue = corr(data1[:,i],data2[:,j])
			if np.isnan(correlation) or np.isnan(pvalue):
				correlation = np.nan
				pvalue = np.nan
			corr_mat[i,j] = correlation
			prob_mat[i,j] = pvalue
	"""
	if do_permute:
		idx1 = range(n1)
		idx2 = range(n2)
		np.random.shuffle(idx1)
		np.random.shuffle(idx2)
		data1_p = data1[idx1,:]
		data2_p = data2[idx2,:]
		count = 0
		for i in itertools.permutations(range(n1), n1):
			idx = np.asarray(i)
			data1_p[count,:] = data1_p[idx,:]
			count += 1
			if count == p1:break
		count = 0
		for i in itertools.permutations(range(n2), n2):
			idx = np.asarray(i)
			data2_p[count,:] = data2_p[idx,:]
			count += 1
			if count == p2:break
	"""
	return corr_mat,prob_mat

def matrix_1Dnarr(mat):
	n,p = mat.shape
	narr = np.asarray(mat.reshape(n*p))[0]
	return narr



if __name__ == "__main__":
	X = np.random.random((25,10000))
	#corr,pvasls = nancorrmatrix_self(X)
	#print corr
	#print pvasls
	nancorrmatrix_self_mp2(X,"tmppath",nprocessor=12)
