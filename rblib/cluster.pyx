import sys
import numpy as np
from sklearn import metrics
from sklearn import cluster


class Cluster(object):
	def __init__(self,Xarray=None,Xdist=None,threads=4,ncluster=50):#,minclustersize=30):
		self.Xarray = Xarray  # n X p
		self.Xdist  = Xdist   # p X p
		self.cluster = None
		self.threads = threads
		self.ncluster = ncluster
	def caldist(self):
		"""
		Valid values for metric are:
		From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]. These metrics support sparse matrix inputs.
		From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’] See the documentation for scipy.spatial.distance for details on these metrics. These metrics do not support sparse matrix inputs.
		"""
		#if instance(self.Xdist,type(None)):
		#if self.Xdist == None:### n X p  => p X p
		#	self.Xdist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(self.Xarray.T,'euclidean')) # distance metric, 
		Xdist = metrics.pairwise.pairwise_distances(self.Xarray,metric="euclidean",n_jobs=self.threads) # 
		return 0
	def hcluster(self):
		"""
		linkage : {“ward”, “complete”, “average”}, optional

		self.cluster:
		labels_ : array [n_samples]
		cluster labels for each point
		n_leaves_ : int
		Number of leaves in the hierarchical tree.
		n_components_ : int
		The estimated number of connected components in the graph.
		children_ : array-like, shape (n_nodes-1, 2)
		The children of each non-leaf node. Values less than n_samples correspond to leaves of the tree which are the original samples. A node i greater than or equal to n_samples is a non-leaf node and has children children_[i - n_samples]. Alternatively at the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i
		"""
		if isinstance(self.Xdist,type(None)): self.caldist()
		clustering = cluster.AgglomerativeClustering(linkage="average", n_clusters=self.ncluster,affinity="precomputed")
		self.cluster = clustering.fit(self.Xdist)
		return self.cluster.labels_ #

