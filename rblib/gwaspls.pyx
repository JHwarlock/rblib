"""
------This is a GWAS method with partial least squares (PLS) of multiple SNPs test.------
	  Here, we use the PLS and Permutation test for VIP Statistics.
	  
	  Programed by Rong Zhenq-Qin (rongzhenqqin@genomics.org.cn  or  zju3351689@gmail.com)
"""
import sys
import numpy as np
import time
import struct
import scipy
import scipy as sp
from scipy.cluster.vq import whiten,kmeans2,kmeans

def resampling1(wvector,numselect):
	"""
	This is a method of resampling with replacement based on the  probability created by weight matrix.
	Here, we used a Cumulative probability method to resampling.

	Input:
		wvector:	the weight vector of samples (is a nx1 list vector)
		numselect:	the number of samples resampling

	Output:
		Index:	the index of resampling samples (a list)
	"""

	wvector = map(float,wvector)
	pro = wvector/np.sum(wvector)
	procum = np.cumsum(pro)
	randseq = np.random.rand(numselect,1)
	Index = []
	for i in xrange(numselect):
		temp = np.random.rand()<procum
		temp = list(temp)
		Index.append(temp.index(True))
	return Index


def resampling2(wvector,numselect):
	"""
	This is a method of resampling without replacement based on the probability created by weight matrix.
	It is similar to resampling with replacement
	It is also could be used to sort randomly (where the wvector is ones vector, and the len(wvector) == numselect)
	
	Input:
		wvector:	the weight vector of samples (is a 1 x n list vector)
		numselect:	the number of samples resampling

	Output:
		Index:	the index of resampling samples (a list)
	"""

	wvector = map(float,wvector)
	pro = wvector/np.sum(wvector)
	procum = np.cumsum(pro)
	randseq = np.random.rand(numselect,1)
	Index = []

	def weightas0(pro,ind,procum):
		"""
		This is a sub-function to be easy to resampling.
		"""

		prosum = procum[-1]-pro[ind]
		pro[ind] = 0;
#		print pro
#		print prosum
		pro = pro/prosum;
		procum = np.cumsum(pro)
		return pro,procum

	for i in xrange(numselect-1):
		temp = np.random.rand()<procum
		temp = list(temp)
		ind = temp.index(True)
		Index.append(ind)
		pro,procum = weightas0(pro,ind,procum)
	i = numselect-1;temp = np.random.rand()<procum;temp = list(temp);ind = temp.index(True);Index.append(ind);
	#print len(Index)
	#print len(set(Index))
	return Index


class output:
	def __init__(self,n,px,nlvs,xtype):
		self.scoreX = np.matrix(np.zeros((n,nlvs)),dtype=xtype)
		self.scoreY = np.matrix(np.zeros((n,nlvs)),dtype=xtype)
		self.W = np.matrix(np.zeros((px,nlvs)),dtype=xtype)
class output_pca:
	def __init__(self,n,px,nlvs,xtype):
		self.scoreX = np.matrix(np.zeros((n,nlvs)),dtype=xtype)
		self.loadings = np.matrix(np.zeros((px,nlvs)),dtype=xtype)
		self.expvars = np.zeros(nlvs)

class lssvmmodel:
	def __init__(self):
		self.b = 0
		self.alpha = 0
		self.yp = 0
		self.kernel_type = ''
		self.parameter = 0
		self.Xtrain = 0
		self.method = ''


class mdsoutput:
	def __init__(self):
		self.w = 0
		self.v = 0 # score
		self.p = 0 # variance
def check_data(X_SNPs):
	if X_SNPs.dtype == np.float64 or X_SNPs.dtype == np.float32:
		xtype = X_SNPs.dtype
		pass
	else:
		sys.stderr.write("""The format of X_SNPs matrix should be numpy.float32 or numpy.float64, please check it.
		If the memory is sufficient, we suggest you use the numpy.float64\n""")
		sys.exit(1)

import mutilstats
def mds_ps(X_raw,nvs_output=10,norm=1):
	"""
	Here, a mutil demensional scale method was used to calculate the population structure.
	I think this must be improve, we will check the reliability of method for population structure analysis, soon.
	
	Input:
		nvs_output: is number of reduced demensions of raw data
		X_SNPs: is the same as it in plsgwas
	Output:
		w : a list (len(list) = nvs_output) of eigenvalue
		v : a matrix of eigenvector corresponds to the eigenvalue
	"""
	X_SNPs = X_raw.copy()
	if X_SNPs.dtype == np.float64 or X_SNPs.dtype == np.float32:
		xtype = X_SNPs.dtype
		pass
	else:
		sys.stderr.write("""The format of X_SNPs matrix should be numpy.float32 or numpy.float64, please check it.
		If the memory is sufficient, we suggest you use the numpy.float64\n""")
		sys.exit(1)
	nx,px = X_SNPs.shape
	X_SNPs = np.asmatrix(X_SNPs)
	if nvs_output>nx:
		sys.stderr.write('too many nvs_output, it must be smaller than number of samples, we have changed auto\n')
	nvs_output = min(nx,nvs_output)
	mutilstats.centring(X_SNPs)
	if norm:
		mutilstats.normalize(X_SNPs)
	#print X_SNPs
	dist = np.asmatrix(np.zeros((nx,nx)))
	for i in xrange(nx):
		temp = X_SNPs - X_SNPs[i,:]
		temp = np.power(temp,2)
		dist[:,i] = np.power(np.sum(temp,axis=1),0.5)
	I = np.asmatrix(np.eye(nx))
	I_n = np.asmatrix(np.ones((nx,nx)))
	dist = -1*(I-(1.0/nx)*I_n)*dist*(I-(1.0/nx)*I_n)/2
	del I_n
	del I
	w,v=np.linalg.eig(dist)
	del dist
	idx = np.argsort(w)[::-1]
	w = w[idx]
	v = v[:,idx]
	
	precent = np.cumsum(w)/np.sum(w) * 100
	print precent
	mds_output = mdsoutput()
	mds_output.p = precent[0:nvs_output]
	mds_output.w = w[0:nvs_output]
	mds_output.v = v[:,0:nvs_output]
	"""
	w=list(w)
	wtemp=w[:]
	wtemp.sort()
	last=-1
	vector_ind = []
	return_v = np.asmatrix(np.zeros((nx,nvs_output)))
	while nvs_output:
		vector_ind.append(w.index(wtemp[last]))
		last -= 1
		nvs_output -= 1
	return_w = []
	while vector_ind:
		ind = vector_ind.pop(0)
		return_w.append(w[ind])
		return_v[:,nvs_output] = v[:,ind]
		nvs_output += 1
	"""
	return mds_output

import scipy as sp

def expvar(error_mat):
	"""call error matrix """
	return np.sum(np.sum(np.asarray(error_mat)**2))

def pca(X_SNPs,nvs_output=10,norm=1):
	"""
	we use the  NIPALS algorithm, the algorithm can be found in http://folk.uio.no/henninri/pca_module/pca_nipals.pdf
	"""
	if X_SNPs.dtype == np.float64 or X_SNPs.dtype == np.float32:
		xtype = X_SNPs.dtype
		pass
	else:
		sys.stderr.write( """
		The format of X_SNPs matrix should be numpy.float32 or numpy.float64, please check it.
		If the memory is sufficient, we suggest you use the numpy.float64\n
		""")
		sys.exit(1)
	nx,px = X_SNPs.shape
	X_SNPs = np.asmatrix(X_SNPs)
	if nvs_output>nx:
		sys.stderr.write("too many nvs_output, it must be smaller than number of samples, we have auto chage it to '%d'\n"%(nx-1))
		nvs_output = nx - 1
	# do mean centre
	Xmean = np.mean(X_SNPs,axis=0)
	X = X_SNPs - Xmean
	# to do normalize 
	if norm:
		Xstd =np.std(X,ddof=1,axis=0)
		X /= Xstd
	#print Xstd
	# eig method 
	"""
	eigvalue,eigvec = sp.linalg.eig(np.cov(X.T))
	idx = np.argsort(eigvalue)[::-1]
	nvs_output = min(nvs_output,len(eigvalue))
	
	eigvalue = eigvalue[idx]
	print eigvalue
	eigvec = eigvec[:,idx]
	print eigvec
	percent = np.cumsum(eigvalue)/np.sum(eigvalue) * 100
	eigvecselect = np.asmatrix(eigvec[:,0:nvs_output])
	scoreselect= X * eigvecselect
	"""
	# NIPALS method 
	result = output_pca(nx,px,nvs_output,xtype)
	t0 = X_SNPs[:,0]
	if t0.T*t0 == 0:
		t0 += 1
	eps = 1e-10
	totalerr = expvar(X_SNPs)
	sys.stderr.write("[INFO] total error: %5.g\n"%totalerr)
	i = 0
	while i<nvs_output:
		maxerror = 1
		iter = 1
		while(i+1):
			if maxerror < eps:
				break
			else:
				pvec = (X_SNPs.T * t0)/(t0.T*t0)
				pvec /= np.linalg.norm(pvec)
				tvec = (X_SNPs*pvec)/(pvec.T*pvec)
				#print tvec.shape,"tv"
				#print pvec.shape,"pv"
				#tvec /= np.linalg.norm(tvec)
				temp = np.abs(tvec-t0)
				maxerror = np.max(temp)
				t0 = tvec
				if iter % 10 == 0: sys.stderr.write("[INFO]: LV %d, Error after %d iterations: %5.g\n"%(i+1,iter,maxerror))
				iter += 1
		result.scoreX[:,i] = tvec
		result.loadings[:,i] = pvec
		X_SNPs = X_SNPs - (tvec*pvec.T)
		t0 = X_SNPs[:,0]
		result.expvars[i] = (totalerr - expvar(X_SNPs))/totalerr*100
		i += 1
	#return percent[0:nmax],eigvecselect,scoreselect
	return result

def fastdist(x,y):
	D = np.sum(np.power(y.T,2),axis=0).T * np.asmatrix(np.ones((1,x.shape[0]))) + np.asmatrix(np.ones((y.shape[0],1))) * np.sum(np.power(x.T,2),axis=0) - 2*(y*x.T)
	return D

def kenstone(X,Y,k,method="raw"):
	model = []
	n,p = X.shape
	if method == "pca":
		npca = 0
		pca_ret = pca(X,n-1)
		for i in xrange(len(pca_ret.expvars)):
			if pca_ret.expvars[i] > 99:
				npca = i
				break
		Xm = pca_ret.scoreX[:,0:i+1]
		
	elif method == "raw":
		Xm = X.copy()
	Xm = np.asmatrix(Xm)
	idx = np.arange(Xm.shape[0])
	idx.shape = Xm.shape[0],1
	Xmm = np.concatenate((idx,Xm),axis=1)
	tmpmean = np.mean(Xm,axis=0)
	tmpidx = np.argmin(fastdist(tmpmean,Xm),axis=0)[0,0]
	model.append(np.int(Xmm[tmpidx,0]))
	Xmm = np.delete(Xmm,tmpidx,axis=0)
	
	tmpidx = np.argmax(fastdist(Xm[model[0],:],Xmm[:,1:]),axis=0)[0,0]
	model.append(np.int(Xmm[tmpidx,0]))
	Xmm = np.delete(Xmm,tmpidx,axis=0)

	for i in xrange(k-2):
		tmpidx=np.argmax(np.min(fastdist(Xmm[:,1:],Xm[model,:]),axis=0),axis=1)[0,0]
		#print fastdist(Xmm[:,1:],Xm[model,:])
		#print np.min(fastdist(Xmm[:,1:],Xm[model,:]),axis=0)
		#print np.argmax(np.min(fastdist(Xmm[:,1:],Xm[model,:]),axis=0),axis=1)
		model.append(np.int(Xmm[tmpidx,0]))
		Xmm = np.delete(Xmm,tmpidx,axis=0)
	others = list(set(range(n)) - set(model))
	return model,others

def plsgwas2(X_SNPs,Y,nlvs=10):
	"""
	This is the main algorithm2 for GWAS based on PLS, which is also suitable for mutile traits.

	Input:
	X_SNPs: is a matrix of n x p (n is the number of samples and p is the number of the SNPs),the format is numpy.matrix (not numpy.array)
	Y: is a matrix of n x m (it is the m traits of n samples), the format is numpy.matrix (not numpy.array)
	nlvs: number of the latent varibles. Here, we give a default value 10, but we really suggest you determin the parameter by prediction.

	Output:
	result is Beta coefficient matrix.
	"""
	check_data(X_SNPs);check_data(Y);
	X_SNPs = np.asmatrix(X_SNPs)
	Y = np.asmatrix(Y)
	nx,dx = X_SNPs.shape
	ny,dy = Y.shape
	if nx == ny:
		pass
	else:
		sys.stderr.write("the sample number between X_SNPs and Y_trait is not equal, pleas check\n")
		sys.exit(1)
	#Xmean = np.mean(X_SNPs,axis=0);Xstd = np.std(X_SNPs,ddof=1,axis=0);
	#Ymean = np.mean(Y,axis=0)
	#X_SNPs -= Xmean; #X_SNPs = X_SNPs/Xstd;
	#Yh = Y.copy();Yh = Yh-Ymean;
	#LoadX = np.asmatrix(np.zeros((dx,nlvs)));
	Cov = Y.transpose()*X_SNPs;
	Cov = np.asmatrix(Cov.transpose())
	LoadY = np.asmatrix(np.zeros((dy,nlvs)));
	statW = np.asmatrix(np.zeros((dx,nlvs)));
	V = np.asmatrix(np.zeros((dx,nlvs)));
	for i in xrange(nlvs):
		ri,si,ci = np.linalg.svd(Cov,full_matrices=True)
		ri = ri[:,0]; si = si[0]; ci=ci[0,:];
		ti = X_SNPs*ri;
		normti = np.linalg.norm(ti)
		ti = ti / normti
		#LoadX[:,i] = np.transpose(ti.transpose()*X_SNPs)
		vi = np.transpose(ti.transpose()*X_SNPs)
		LoadY[:,i] = np.asmatrix(si*(ci.transpose())/normti)
		statW[:,i] = np.asmatrix(ri / normti)
		#vi = LoadX[:,i];
		for rep in xrange(2):
			for j in xrange(i):
				vj = V[:,j]
				temp = vj.transpose()*vi
				vi = vi - (temp[0,0])*vj
		vi /= np.linalg.norm(vi)
		V[:,i] = vi
		Cov -= (vi*(vi.transpose()*Cov))
		Cov -= (V[:,0:i+1]*(V[:,0:i+1].transpose()*Cov))
	#Beta = statW*LoadY.transpose()
	return statW*LoadY.transpose()

def plsgwas(X_SNPs,Y,nlvs=10):
	"""
	This is the main algorithm for GWAS based on PLS
	
	Input:
		X_SNPs: is a matrix of n x p (n is the number of samples and p is the number of the SNPs),the format is numpy.matrix (not numpy.array)
		Y: is a matrix of n x 1 (it is the trait of n samples), the format is numpy.matrix (not numpy.array)
		nlvs: number of the latent varibles. Here, we give a default value 10, but we really suggest you determin the parameter by prediction

	Output:
		result is class object, which contain three matrix named result.scoreX, result.scoreY, and result.W
		result.scoreX: a matrix of n x nlvs
		result.scoreY: a matrix of n x nlvs
		result.W: a matrix of p x nlvs
		
		This output class obj can be the input of the getVIP function following.
	"""
	if X_SNPs.dtype == np.float64 or X_SNPs.dtype == np.float32:
		xtype = X_SNPs.dtype
		pass
	else:
		sys.stderr.write("""The format of X_SNPs matrix should be numpy.float32 or numpy.float64, please check it\n""")
		sys.exit(1)
	if Y.dtype == np.float64 or Y.dtype == np.float32:
		ytype = Y.dtype
		pass
	else:
		sys.stderr.write("""The format of Y_trait matrix should be numpy.float32 or numpy.float64, please check it.
		If the memory is sufficient, we suggest you use the numpy.float64\n""")
		sys.exit(1)
	
	nx,px = X_SNPs.shape
	#print X_SNPs
	ny,py = Y.shape
	if nx == ny:
		pass
	else:
		sys.stderr.write("the sample number between X_SNPs and Y_trait is not equal, pleas check\n")
		sys.exit(1)
	Yh = Y.copy()
	u0 = Y.copy()
	eps = 1e-10
	result = output(nx,px,nlvs,xtype)
	i = 0;
	while i<nlvs:
		maxerror = 1
		while (i+1):
			if maxerror < eps:
				break
			else:
				w = u0.transpose()*X_SNPs
				w = w.transpose()
				t = X_SNPs*w
				t /= np.linalg.norm(t)
				c = Yh.transpose()*t;
				u = Yh*c
				u /= np.linalg.norm(u)
				temp = np.abs(u-u0)
				maxerror = np.max(temp)
				#print maxerror,i
				u0 = u
		result.scoreX[:,i] = t
		result.scoreY[:,i] = u
		result.W[:,i] = w/np.linalg.norm(w)
		X_SNPs = (np.eye(nx)-t*t.transpose())*X_SNPs
		Yh = Yh-t*t.transpose()*Yh
		u0 = Yh
		i += 1
	#print X_SNPs
	#print result.scoreX
	return result

def getVIP(result,Y):
	"""
	Here is to calculate the statistics of variable important projection (VIP for each SNPs)
	"""
	
	p,nlvs = result.W.shape
	corrlist = []
	for i in xrange(nlvs):
		corrlist.append(np.corrcoef(result.scoreX[:,i].transpose(),Y.transpose())[0][1]**2)
	corrlist = np.asmatrix(corrlist,dtype=result.scoreX.dtype)
	corrlist = corrlist.transpose()
	sumcorr = sum(corrlist)
	W = np.asarray(result.W)
	W2 = W**2
	W2 = np.asmatrix(W2)
	VIPmat = np.sqrt(p*(W2*corrlist)/sumcorr);
	return VIPmat


def randomsort(len_seq):
	wvector = np.ones(len_seq)
	ind = resampling2(wvector,len_seq)
	return ind

def VIPoutput(VIPmat,fh_vip,p):
	structtype = str(p)+'d'
	temp = VIPmat[:,0].transpose().tolist()[0]
	fh_vip.write(struct.pack(structtype,*temp))
	#print "Statics:"
	#print VIPmat
	return 0

def VIPtest(X_SNPs,Y,fh_per,nlvs=10,ntimes=1000):
	"""
	Here, the Permutation test is adopted based on the bootstrap without replacement. It is implemented to obtain the empirical distribution of VIP.
	"""
	Yt = Y.copy()
	p = X_SNPs.shape[-1]
	structtype = str(p)+'d'
	#h = np.zeros((p,1))
	#tmp_file = file('/ifs1/STT_POP/USER/rongzhq/pythread/temp.txt','w+')
	#temp = VIPmat[:,0].transpose().tolist()[0]
	#fh_vip.write(struct.pack(structtype,*temp))
	#print "Statics:"
	#print VIPmat
	for i in xrange(ntimes):
		ind = randomsort(len(Yt))
		Yt = Yt[ind,:]
		result = plsgwas(X_SNPs,Yt,nlvs)
		Viptemp = getVIP(result,Yt)
		sys.stderr.write("Statics_per\t"+str(i+1)+"\n")
		#print Viptemp
		#t1 = time.time()
		#print "ok"
		temp = Viptemp[:,0].transpose().tolist()[0]
		fh_per.write(struct.pack(structtype,*temp))
		#print time.time()-t1
	return 0

def BETAtest(X_SNPs,Y,BETA,fh_beta,fh_per,nlvs=10,ntimes=1000):
	"""
	Here, the Permutation test is adopted based on the bootstrap without replacement. It is implemented to obtain the empirical distribution of BETA
	"""
	Yt = Y.copy()
	n = Yt.shape[0]
	dy = Yt.shape[-1]
	dx = X_SNPs.shape[-1]
	structtype = str(dx)+'d'
	for j in xrange(dy):
		temp = BETA[:,j].transpose().tolist()[0]
		fh_beta.write(struct.pack(structtype,*temp))
	print 'BETA:'
	print BETA
	fh_beta.write(struct.pack(structtype,*temp))
	for i in xrange(ntimes):
		ind = randomsort(n)
		Yt = Yt[ind,:]
		BETA_per = plsgwas2(X_SNPs,Yt,nlvs)
		print 'BETA_per\t'+str(i+1)
		print BETA_per
		for j in xrange(dy):
			temp = BETA_per[:,j].transpose().tolist()[0]
			fh_per.write(struct.pack(structtype,*temp))
	return 0


def MCVIPtest(X_SNPs,Y,fh_vip,fh_per,nlvs=10,ntimes=1000,nsel=200):
	"""
	In order to overcome the bias and enhance the robustness for VIP estimation, the Monte Carlo procedure is implemented to obtain the empirical distribution of VIP. But this method may not be suitable for large scale data, because of the cost of memory. So, we suggest that to use 40%~50% raw data to do it(set the input: nsel = 40%~50% number of individuals)
	"""

	if nsel < X_SNPs.shape[0]:
		pass
	else:
		sys.stderr.write('nsel must be less than number of individuals\n')
		sys.exit(1)
	p = X_SNPs.shape[1]
	#h = np.zeros((p,1))
	for i in xrange(ntimes):
		X0 = X_SNPs.copy()
		Y0 = Y.copy()
		ind = randomsort(len(Y0))
		Y0 = Y0[ind[1:nsel+1],:]
		X0 = X0[ind[1:nsel+1],:]
		result = plsgwas(X0,Y0,nlvs)
		Vipmat = getVIP(result,Y0)
		for j in xrange(p):
			fh_vip.write(struct.pack('d',Vipmat[j,0]))
		ind = randomsort(len(Y0))
		Y0 = Y0[ind,:]
		result = plsgwas(X0,Y0,nlvs)
		Viptemp = getVIP(result,Y0)
		t1 = time.time()
		#print "ok"
		for j in xrange(p):
			fh_per.write(struct.pack('d',Viptemp[j,0]))
		#print time.time()-t1
	return 0




"""
------Here we give a LSSVM for classify and function estimate------
	in the classifier, this LSSVM used for pattern recognition
	in the function estimation, this LSSVM could be for the regression prediction
	Programed by Rong Zhenq-Qin (rongzhq@genomics.org.cn)
"""


def dotvector(X,Xt):
	return Xt*X.transpose()


def rbfnorm2(X,Xt):
	n0 = X.shape[0]
	nx = Xt.shape[0]
	RBFnorm2 = np.zeros((nx,n0));
	for i in xrange(nx):
		for j in xrange(n0):
			RBFnorm2[i,j] = np.linalg.norm(Xt[i,:]-X[j,:])**2
	return RBFnorm2


def kernel(Xtrain,Xtest,kernel_type='rbf',parameter_kernel=1):
	"""
	Here we give four functions, named as linear('lin'), polynomial('poly'), rbf('rbf') and sigmoid('sig')
	default is 'rbf'

	the parameter_kernel is the parameter of kernel function:
	'linear'     -->no parameter_kernel
	'polynomial' -->degree
	'rbf'        -->sig2
	'sigmoid'    -->v_value
	"""

	X = Xtrain.copy()
	Xt = Xtest.copy()
	if kernel_type.lower() == 'rbf':
		sig2 = parameter_kernel
		Kt = np.exp((-1)*rbfnorm2(X,Xt)/sig2)
	elif kernel_type.lower() == 'lin' or kernel_type.lower() == 'linear':
		Kt = dotvector(X,Xt)
	elif kernel_type.lower() == 'poly' or kernel_type.lower() == 'polynomial':
		degree = parameter_kernel
		Kt = np.asmatrix(np.asarray(dotvector(X,Xt))**degree)
	elif kernel_type.lower() == 'sig' or kernel_type.lower() == 'sigmoid':
		v_value = parameter_kernel
		Kt = np.tanh(v_value*dotvector(X,Xt))
	else:
		sys.stderr.write("Unknown kernel function, please check\n")
		sys.exit(1)
	Kt = Kt.transpose()
	return Kt


def bic_kmeans(X,maxk=None):
	"""
	use BIC max for kmeans
	"""
	n,p = X.shape
	if maxk is None:
		maxk = int(np.sqrt(n))
	xy = whiten(X)
	assert maxk+1>2
	for numk in xrange(2,maxk+1):
		#variance_total = np.sum((xy - np.mean(xy,axis=0))**2) / (n-numk)
		#print variance_total
		#res, idx = kmeans2(xy,numk,minit='points')
		res, idx = kmeans2(xy,numk)
		LDtot = 0
		others = numk+numk*p / 2 * np.log(n)
		for i in xrange(numk):
			Rn = np.sum(idx==i)
			if Rn <= 1:continue
			#logRn = np.log(Rn)
			variance = np.sum((xy[idx==i,:] - res[i,:])**2) / (Rn-1) ## 
			LDn = -Rn/2.0 * np.log(2*np.pi) - Rn * p /2.0 *np.log(variance) -(Rn-numk)/2.0 +Rn *np.log(Rn) -Rn *np.log(n)
			LDtot += LDn
		bicvalue = LDtot - others
		print numk,bicvalue,LDtot,others
		#print idx
	return None


def kmeans_cluster(X,numk=3):
	"""
	do numk kmeans
	"""
	xy = whiten(X)
	res, idx = kmeans2(xy,numk)
	return res,idx

def lssvm(X,Y,gamma=1,method='quantificat',kernel_type='rbf',parameter=1):
	"""
	This is a LSSVM for function estimates (regress) and pattern recognition (classify).
	
	Input:
		X: a matrix of (n x p), where n is sample numbers and p is the number of variables
		Y: a matrix of (n x 1), where Y is trait
		two methods:
			'quantificat': means regress or function estimates
			'category': means classify or pattern recognition
		gamma: is the Penalty parameter of SVM, which need to be optimized
		kernel_type: Here we give four functions, named as linear('lin'), polynomial('poly'), rbf('rbf') and sigmoid('sig')
		parameter is the parameter of kernel function:
			'linear'     -->no parameter_kernel
			'polynomial' -->degree
			'rbf'        -->sig2
			'sigmoid'    -->v_value
	
	Output:
		model: is a instance of lssvmmodel class obj, which is used for prediction
	"""

	n = X.shape[0]
	ny = Y.shape[0]
	if n != ny:
		sys.stderr.write('please check the demensions of X and Y\n')
		sys.exit(1)
	Ktrain = kernel(X,X,kernel_type,parameter)
	Ktol = np.zeros((n+1,n+1))
	Ytol = np.zeros((n+1,1))
	if method == 'quantificat' or method == 'regress':
		Ktol[0,1:] = 1
		Ktol[1:,0] = 1
		Ktol[1:,1:] = Ktrain+(1.0/gamma)*np.eye(n)
		Ytol[1:] = Y
		beta = np.linalg.solve(Ktol,Ytol)
		model = lssvmmodel()
		model.b = beta[0]
		model.alpha = np.asmatrix(beta[1:])
		model.kernel_type = kernel_type
		model.parameter = parameter
		model.Xtrain = X.copy()
		model.yp = Ktrain*model.alpha+model.b*np.ones((n,1))
		model.method = 'quantificat'
	elif method == 'category' or method == 'classify':
		Ktol[0,1:] = -1*Y.transpose()
		Ktol = np.asmatrix(Ktol)
		Ktol[1:,0] = Y.copy()
		YY = np.asarray(Y*Y.transpose())
		Ktol[1:,1:] = np.asmatrix(YY*np.asarray(Ktrain))+(1.0/gamma)*np.eye(n)
		Ytol[1:] = np.ones((n,1))
		beta = np.linalg.solve(Ktol,Ytol)
		model = lssvmmodel()
		model.b = beta[0]
		model.alpha = np.asmatrix(beta[1:]*np.asarray(Y))
		model.kernel_type = kernel_type
		model.parameter = parameter
		model.Xtrain = X.copy()
		model.yp = np.sign(Ktrain*model.alpha+model.b*np.ones((n,1)))
		model.method = 'category'
	del Ktol
	return model


class simlssvmresult:
	def __init__(self):
		self.method = ''
		self.ypredict = 0
		self.risk = 10000


def simlssvm(Xtest,model,Ytest=0):
	"""
	Simulate lssvm
	"""
	Xtest = np.matrix(Xtest)
	nx = Xtest.shape[0]
	Kt = kernel(model.Xtrain,Xtest,model.kernel_type,model.parameter)
	simlssvmoutput = simlssvmresult()
	simlssvmoutput.ypredict = Kt.transpose()*model.alpha+model.b*np.ones((nx,1))
	if model.method == 'quantificat':
		simlssvmoutput.method = 'quantificat'
	elif model.method == 'category':
		simlssvmoutput.method = 'category'
		simlssvmoutput.ypredict = np.sign(simlssvmoutput.ypredict)
	else:
		sys.stderr.write('There is an unknown method\n')
		sys.exit(1)
	if len(Xtest) != len(Ytest):
		sys.stderr.write('no result in this, because of no Ytest input or sample numbers in X and Y are different\n')
	else:
		if model.method=='quantificat':
			simlssvmoutput.risk = np.sqrt(np.sum((np.asarray(Ytest-simlssvmoutput.ypredict))**2)/Xtest.shape[0])
		else:
			simlssvmoutput.risk = (len(Ytest)-np.sum(np.abs(Ytest-simlssvmoutput.ypredict))/2.0)/len(Ytest)*100.0
	return simlssvmoutput


def crossvalidate(Xtrain,Ytrain):
	"""
	Data sorted randomly for crossvalidation
	"""
	X = Xtrain.copy()
	Y = Ytrain.copy()
	len_seq=len(Y)
	ind = randomsort(len_seq)
	X = X[ind,:]
	Y = Y[ind,:]
	return X,Y

def plsmcuve(X,Y,nlvs=10,ratio=0.5,bootstrap_num=200):
	#ratio is from 0~1, default = 0.5
	nx,px = X.shape
	IND = np.random.rand(bootstrap_num,nx) <= ratio
	BETA = np.asmatrix(np.zeros((px,bootstrap_num)))
	for i in xrange(bootstrap_num):
		betatmp = plsgwas2(X[IND[i,:],:],Y[IND[i,:],:],nlvs)
		BETA[:,i] = betatmp[:,0]
	betamean = np.mean(BETA,axis=1)
	betastd = np.std(BETA,ddof=1,axis=1)
	stability = betamean/betastd
	return stability

def plsmcvip(X,Y,nlvs=10,ratio=0.5,bootstrap_num=200):
	nx,px = X.shape
	IND = np.random.rand(bootstrap_num,nx) <= ratio
	VIPs = np.asmatrix(np.zeros((px,bootstrap_num)))
	for i in xrange(bootstrap_num):
		result = plsgwas(X[IND[i,:],:],Y[IND[i,:],:],nlvs)
		VIPtmp = getVIP(result,Y[IND[i,:],:])
		#print np.sum(np.power(VIPtmp,2))
		VIPs[:,i] = VIPtmp[:,0]
	VIPs = np.log(VIPs)
	VIPmean = np.mean(VIPs,axis=1)
	VIPstd = np.std(VIPs,ddof=1,axis=1)
	vipstability = VIPmean/VIPstd
	return vipstability

def centring(X,axis=0,Xmean = None):
	ret = None
	if Xmean is None:
		Xmean = np.mean(X,axis=axis)
	X -= Xmean
	return Xmean

def normalize(X,axis=0,Xstd=None):
	ret = None
	if Xstd is None:
		Xstd =np.std(X,ddof=1,axis=axis)
	X /= Xstd
	return Xstd

def pls_validataion_risk(X0,Y0,Xmean,Xstd,Ymean,Ystd,Xtest,Ytest,nlvs,method='quantificat',retpred=0):
	if method != 'quantificat' and method != 'category':
		sys.stderr.write("Cross Validation Error:\n  Unrecognized Methods: "+method+"\n")
		return None
	n,p = X0.shape
	nx,px = Xtest.shape

	betatmp = plsgwas2(X0,Y0,nlvs)
	Xtest_scale = Xtest.copy()
	centring(Xtest_scale,Xmean=Xmean)
	normalize(Xtest_scale,Xstd=Xstd)

	ypredict_raw = np.dot(Xtest_scale,betatmp)
	ypredict = ypredict_raw * Ystd + Ymean

	if method == "quantificat":
		risk = np.sqrt(np.sum(np.asarray(Ytest - ypredict)**2)/nx)
	else:
		risk = np.sum(np.sign(ypredict_raw) == np.sign((Ytest - Ymean))) / float(nx)
	if retpred:
		return risk,ypredict
	else:
		return risk

def pls_calibration_risk(X0,Y0,nlvs = 10,method='quantificat',retpred=0):
	if method != 'quantificat' and method != 'category':
		sys.stderr.write("Cross Validation Error:\n  Unrecognized Methods: "+method+"\n")
		return None
	n,p = X0.shape
	betatmp = plsgwas2(X0,Y0,nlvs)
	ypredict = X0*betatmp
	if method == 'quantificat':
		Yfortest = Y0-ypredict
		risk = np.sqrt(np.sum(np.asarray(Yfortest)**2)/n)
	else:
		Yfortest = (np.sign(Y0) == np.sign(ypredict))
		risk = np.sum(Yfortest)/float(n)
	if retpred:
		return risk,ypredict
	else:
		return risk

def plscv(X0,Y0,nlvs = 10,nfold=4,method='quantificat'):
	#plsgwas2(X_SNPs,Y,nlvs=10)
	if method != 'quantificat' and method != 'category':
		sys.stderr.write("Cross Validation Error:\n  Unrecognized Methods: "+method+"\n")
		return None
	Xcv,Ycv = crossvalidate(X0,Y0)
	if method == 'category':
		Ycv = np.sign(Ycv)
	len_seq = len(Ycv)
	Yfortest = Ycv.copy()
	Ycv = np.float64(Ycv)
	block_size = int(len_seq/nfold)
	for i in xrange(nfold):
		l = i+1
		if l == nfold:
			train = range(0,block_size*(l-1))
			validation = range(block_size*(l-1),len_seq)
		else:
			train = range(0,block_size*(l-1))+range(block_size*l,len_seq)
			validation = range(block_size*(l-1),block_size*l)
		Xtraintemp = Xcv[train,:]
		
		Xtraintempmean = centring(Xtraintemp)
		Xtraintempstd = normalize(Xtraintemp)
		
		Ytraintemp = Ycv[train,:]
		Ytesttemp = Ycv[validation,:]
		
		#if method =='quantificat':
			#Ytraintempmean = centring(Ytraintemp)
			#Ytraintempstd = normalize(Ytraintemp)
			#centring(Ytesttemp,Xmean=Ytraintempmean)
			#normalize(Ytesttemp,Xstd=Ytraintempstd)
		
		Xtesttemp = Xcv[validation,:]
		
		centring(Xtesttemp,Xmean=Xtraintempmean)
		normalize(Xtesttemp,Xstd = Xtraintempstd)
		
		betatmp = plsgwas2(Xtraintemp,Ytraintemp,nlvs)
		ypredict = Xtesttemp*betatmp

		if method == 'quantificat':
			Yfortest[validation,:] = Yfortest[validation,:]-ypredict
			#print Yfortest[validation,:]
		else:
			Yfortest[validation,:] = (np.sign(Yfortest[validation,:]) == np.sign(ypredict))
	if method == 'quantificat':
		risk = np.sqrt(np.sum(np.asarray(Yfortest)**2)/len_seq)
	else:
		risk = np.sum(Yfortest)/float(len_seq)
	return risk

def lssvmcv(X,Y,nfold=4,gamma=1,method='quantificat',kernel_type='rbf',parameter=1):
	"""
	This is a n-fold cross validation for LS-SVM
	
	output:
		risk: is Classification accuracy (for pattern recognition)
			  is RMSECV (Root mean square error of cross validation) (for function estimation)
	"""

	Xcv,Ycv = crossvalidate(X,Y)
	len_seq = len(Ycv)
	Yfortest = Ycv.copy()
	block_size = int(len_seq/nfold)
	for i in xrange(nfold):
		l = i+1
		if l == nfold:
			train = range(0,block_size*(l-1))
			validation = range(block_size*(l-1),len_seq)
		else:
			train = range(0,block_size*(l-1))+range(block_size*l,len_seq)
			validation = range(block_size*(l-1),block_size*l)
		Xtraintemp = Xcv[train,:]
		Ytraintemp = Ycv[train,:]
		Xtesttemp = Xcv[validation,:]
		Ytesttemp = Ycv[validation,:]
		modeltemp = lssvm(Xtraintemp,Ytraintemp,gamma,method,kernel_type,parameter)
		outputtemp = simlssvm(Xtesttemp,modeltemp,Ytesttemp)
		outputtemp.ypredict
		Yfortest[validation,:]=Yfortest[validation,:]-outputtemp.ypredict
	if outputtemp.method == 'quantificat':
		risk = np.sqrt(np.sum(np.asarray(Yfortest)**2)/len_seq)
	elif outputtemp.method == 'category':
		risk = (len_seq-np.sum(np.abs(Yfortest))/2.0)/len_seq*100.0
	return risk



if __name__=='__main__':
	"""
	#print "This is a GWAS Algorithm based on PLS and Permutation test"
	#w=np.array([1,1,1,1,1,1000,1])
	#ind=resampling2(w,6)
	#while ind:
		#print ind.pop(0)
	X = np.matrix([[1,2,3,4,5],[2,3,4,5,6],[2,3,4,7,8],[1,6,3,2,4]],dtype=np.float64)
	Y = np.matrix([1,2,5,3,7],dtype=np.float64)
	Y = np.matrix([-1,1,-1,-1,1],dtype=np.float64)
	Xt = np.matrix([[1,3,4],[2,6,1],[5,1,8],[6,3,0]],dtype=np.float64)
	Xt = Xt.transpose()
	X = X.transpose()
	Y = Y.transpose()
	#result = plsgwas(X,Y,nlvs=3)
	#print 'ok'
	#print result.scoreX
	#print result.scoreY
	#print result.W
	model = lssvm(X,Y,1,'category','rbf',1)
	print model.yp
	print Y
	simlssvmoutput = simlssvm(X,model,Y)
	print simlssvmoutput.ypredict
	print simlssvmoutput.risk
	x=np.asmatrix(np.zeros((1000,2)))
	x1=np.random.rand(500,2)
	x2=np.random.rand(500,2)*(-1)
	x[0:500,:]=x1
	x[500:,:]=x2
	result=mds_ps(x,3)
	print 'ok'
	fh = file('test_mds','w')
	for i in xrange(1000):
		fh.write(str(result.v[i,0])+'\t'+str(result.v[i,1])+'\n')
	X = np.matrix([[1,2,3,4],[3,4,5,7],[19,11,12,14]],dtype=np.float64)
	Y = np.matrix([[5,11,2],[2,3,5]],dtype = np.float64)
	Y = Y.transpose()
	Beta = plsgwas2(X,Y,2)
	print Beta
	"""
	pt1 = np.random.normal(1, 0.2, (100,2000))
	pt2 = np.random.normal(50, 0.2, (100,2000))
	pt3 = np.random.normal(100, 0.2, (100,2000))
	xy = np.concatenate((pt1, pt2, pt3))
	bic_kmeans(xy,10)

