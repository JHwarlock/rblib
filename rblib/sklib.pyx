import sys
import numpy as np
from scipy.cluster.vq import whiten
from sklearn import cluster, datasets
from scipy import stats

def bic_kmeans(X,maxk=None):
	"""
	use BIC max for kmeans
	"""
	n,p = X.shape
	if maxk == None:
		maxk = int(np.sqrt(n))
	xy = whiten(X)
	assert maxk+1>2
	retbicvalues = []
	reses = []
	idxes = []
	for numk in range(1,maxk+1):
		#variance_total = np.sum((xy - np.mean(xy,axis=0))**2) / (n-numk)
		#print variance_total
		#res, idx = kmeans2(xy,numk,minit='points')
		res, idx = kmeans(xy,numk)
		reses.append(res)
		idxes.append(idx)
		LDtot = 0
		others = numk+numk*p / 2 * np.log(n)
		for i in range(numk):
			Rn = np.sum(idx==i)
			if Rn <= 1:continue
			#logRn = np.log(Rn)
			variance = np.sum(np.asarray(xy[idx==i,:] - res[i,:])**2) / (Rn-1) ## 
			LDn = -Rn/2.0 * np.log(2*np.pi) - Rn * p /2.0 *np.log(variance) -(Rn-numk)/2.0 +Rn *np.log(Rn) -Rn *np.log(n)
			LDtot += LDn
		bicvalue = LDtot - others
		retbicvalues.append(bicvalue)
		sys.stderr.write("For %d clusters, BIC value is %.2f\n"%(numk,bicvalue))
	#print idx
	select_k = retbicvalues.index(max(retbicvalues)) + 1
	return select_k,reses[select_k-1],idxes[select_k-1]

def kmeans(X,numk=3,nthreads=2):
	k_means = cluster.KMeans(n_clusters=numk,n_jobs=nthreads)
	k_means.fit(X)
	return k_means.cluster_centers_,k_means.labels_


def proportion_test(x1,n1,x2,n2,d0=0,tail="two"):
	p1 = x1*1.0/n1
	p2 = x2*1.0/n2
	p = (x1+x2)*1.0/(n1+n2)
	if d0 == 0:
		z = (p1-p2)/np.sqrt(p*(1-p)*(1.0/n1+1.0/n2))  # H0  z= 0?
	else:
		z = (p1-p2-d0)/np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2) # z = d0 ?
	left = stats.norm.cdf(z)
	right =  1 - stats.norm.cdf(z)

	if tail == "left":
		return  stats.norm.cdf(z)
	elif tail == "right":
		return  1 - stats.norm.cdf(z)
	else:
		return (1-stats.norm.cdf(np.abs(z))) *2

if __name__ == "__main__":
	"""
	iris = datasets.load_iris()
	X_iris = iris.data
	print type(X_iris)
	print X_iris.shape
	y_iris = iris.target
	k_means = cluster.KMeans(n_clusters=3)
	k_means.fit(X_iris)
	print k_means.cluster_centers_
	print k_means.labels_
	print y_iris
	#print cluster.KMeans.__doc__
	pt1 = np.random.normal(1, 0.2, (100,2000))
	pt2 = np.random.normal(50, 0.2, (100,2000))
	pt3 = np.random.normal(100, 0.2, (100,2000))
	xy = np.concatenate((pt1, pt2, pt3))
	print bic_kmeans(xy,maxk=60)
	#print cents,labels
	"""
	print(proportion_test(5.0,25.0,8.0,17.0,tail='left'))

