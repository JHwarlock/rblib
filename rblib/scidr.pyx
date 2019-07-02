import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise


def doPCoA(Xnp,usedist='braycurtis',n_components=3,metric=True,eps=1e-8,n_jobs=1,random_state=None,dissimilarity='euclidean',n_init=4,max_iter=300,verbose=0,precompute=0):
	if not precompute:
		Xdist = pairwise.pairwise_distances(Xnp,metric="braycurtis",n_jobs=n_jobs)
	else:
		assert Xnp.shape[0] == Xnp.shape[1]
		Xdist = Xnp
	Xscore = doMDS(Xdist,n_components=n_components,metric=metric,n_init=n_init,max_iter=max_iter,verbose=verbose,eps=eps,n_jobs=n_jobs,random_state=random_state,dissimilarity="precomputed")
	return Xscore

def doMDS(Xnp,n_components=3,metric=True,eps=1e-8,n_jobs=1,random_state=None,dissimilarity='euclidean',n_init=4,max_iter=300,verbose=0):
	mdsmodel = MDS(n_components=n_components,metric=metric,n_init=n_init,max_iter=max_iter,verbose=verbose,eps=eps,n_jobs=n_jobs,random_state=random_state,dissimilarity=dissimilarity)
	Xscore = mdsmodel.fit_transform(Xnp)
	Xdist = pairwise.pairwise_distances(Xnp,metric="euclidean",n_jobs=n_jobs)
	#nx,p = Xnp.shape
	#I = np.asmatrix(np.eye(nx))
	#I_n = np.asmatrix(np.ones((nx,nx)))
	#dist = -1*(I-(1.0/nx)*I_n)*Xdist*(I-(1.0/nx)*I_n)/2
	#w,v=np.linalg.eig(dist)
	#print w
	#print "##",v
	#print np.cumsum(w)/np.sum(w) * 100
	#print w/np.sum(w) * 100
	return Xscore,mdsmodel.stress_


def doPCA(Xnp,n_components=3,whiten=False,copy=True):
	pcamodel = PCA(n_components=n_components,whiten=whiten,copy=copy)
	pcamodel.fit(Xnp)
	#print pcamodel.components_.shape
	#print pcamodel.explained_variance_ratio_
	Xscore = pcamodel.fit_transform(Xnp)
	EVR = pcamodel.explained_variance_ratio_ * 100
	return Xscore,EVR

if __name__ == "__main__":
	import numpy as np
	np.random.seed(0)
	X = np.random.random((10,5))
	X[0:5,:] += 5
	print X
	#print doPCA(X,whiten = True)
	
	### MDS varaince 
	#Xscore,stress = doMDS(X,n_components=2)
	#xx=  np.sum(Xscore ** 2,axis=0)   ## axis is not sort, and for sum of squares,  n_components = 2  is equal to n_components = 5
	#print xx
	#print np.sum(xx)
	#print stress

