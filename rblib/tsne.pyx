#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as Math
import matplotlib
matplotlib.use("Agg")
import pylab as Plot
import sys
from rblib import mutilstats

def Hbeta(D = Math.array([]), beta = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	P = Math.exp(-D.copy() * beta);
	sumP = sum(P);
	if sumP <=0:
		sumP = 1e-100
	#print sumP
	H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;


def x2p(X = Math.array([]), tol = 1e-5, perplexity = 30.0):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	sys.stderr.write("Computing pairwise distances...\n")
	(n, d) = X.shape;
	sum_X = Math.sum(Math.square(X), 1);
	D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X);
	P = Math.zeros((n, n));
	beta = Math.ones((n, 1));
	logU = Math.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			sys.stderr.write(" ".join(["Computing P-values for point ", str(i), " of ", str(n), "...\n"]))

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -Math.inf;
		betamax =  Math.inf;
		Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while Math.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == Math.inf or betamax == -Math.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == Math.inf or betamin == -Math.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;

	# Return final P-matrix
	sys.stderr.write("Mean value of sigma: %.3g\n"%(Math.mean(Math.sqrt(1 / beta))));
	return P;


def pca(X = Math.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	sys.stderr.write("Preprocessing the data using PCA...\n")
	(n, d) = X.shape;
	X = X - Math.tile(Math.mean(X, 0), (n, 1));
	(l, M) = Math.linalg.eig(Math.dot(X.T, X));
	Y = Math.dot(X, M[:,0:no_dims]);
	return Y;


def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
	"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

	# Check inputs
	if isinstance(no_dims, float):
		sys.stderr.write("Error: array X should have type float.\n");
		return -1;
	if round(no_dims) != no_dims:
		sys.stderr.write("Error: number of dimensions should be an integer.\n");
		return -1;

	# Initialize variables
	X = pca(X, initial_dims).real;
	#Plot.scatter(X[:,0], X[:,1], 20, labels);
	Plot.savefig("test.pca.png",format='png',dpi=300)
	Plot.clf();Plot.close()
	(n, d) = X.shape;
	max_iter = 1000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = Math.random.randn(n, no_dims);
	dY = Math.zeros((n, no_dims));
	iY = Math.zeros((n, no_dims));
	gains = Math.ones((n, no_dims));

	# Compute P-values
	P = x2p(X, 1e-5, perplexity);
	P = P + Math.transpose(P);
	P = P / Math.sum(P);
	P = P * 4;									# early exaggeration
	P = Math.maximum(P, 1e-12);

	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = Math.sum(Math.square(Y), 1);
		num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[list(range(n)), list(range(n))] = 0;
		Q = num / Math.sum(num);
		Q = Math.maximum(Q, 1e-12);

		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = Math.sum(P * Math.log(P / Q));
			sys.stderr.write("Iteration %d: error is %.5g\n"%(iter + 1,C))

		# Stop lying about P-values
		if iter == 100:
			P = P / 4;

	# Return solution
	return Y;

"""
def run_tsne(X,Y,no_dims = 2, initial_dims = 50, perplexity = 30.0,figprefix="test_tsne",markersize=20,alpha=0.5):
	mutilstats.centring(X)
	labels = Y
	newX = tsne(X, no_dims, initial_dims, perplexity);
	Plot.scatter(newX[:,0], newX[:,1], markersize, labels,alpha=0.5);
	Plot.savefig("%s.png"%figprefix,format='png',dpi=300)
	Plot.savefig("%s.svg"%figprefix,format='svg',dpi=300)
	Plot.clf();Plot.close()
	return 0
"""


### new version to process
import numpy as np
from sklearn.decomposition import PCA
from rblib import statplot,gwaspls
from sklearn.manifold import TSNE

def tsne2(X,Y,ntsne=2,ndim=10,method="raw",plotprefix="tnse-plot-default",verbose=1,perplexity=40,n_iter=300):
	"""  X is n X p array, and Y is n vector or list"""
	assert method in ["raw","classic-pca","nipals-pca","mds"]
	n,p = X.shape
	assert n > ndim and p > ndim
	if isinstance(Y,list): labels = Y
	else:
		labels = Y.tolist()
	statplot.groups_scatter_flatdata(X[:,0],X[:,1],labels,"Vector 1","Vector 2",fig_prefix=plotprefix+".dist")
	if method == "raw":pass
	elif method == "classic-pca":
		pca = PCA(n_components=ndim)
		pca_result = pca.fit_transform(X)
		sys.stderr.write('Explained variation per principal component: {} \n'.format(pca.explained_variance_ratio_))
		X = pca_result[:,0:ndim]
		statplot.groups_scatter_flatdata(X[:,0],X[:,1],labels,"Eigen vector 1","Eigen vector 2",fig_prefix=plotprefix+".classic-pca")
	elif method == "nipals-pca":
		output = gwaspls.pca(X,nvs_output=ndim,norm=0) # norm = 0 means only mean center process
		sys.stderr.write('Explained variation per principal component: {} \n'.format(output.expvars))
		X = np.asarray(output.scoreX[:,0:ndim])
		statplot.groups_scatter_flatdata(X[:,0],X[:,1],labels,"Eigen vector 1","Eigen vector 2",fig_prefix=plotprefix+".nipals-pca")
	elif method == "mds":
		output = gwaspls.mds_ps(X,nvs_output=ndim,norm=0)
		sys.stderr.write('Explained variation per principal component: {} \n'.format(output.p))
		X = np.asarray(output.v[:,0:ndim])
		statplot.groups_scatter_flatdata(X[:,0],X[:,1],labels,"Dimension 1","Dimension 2",fig_prefix=plotprefix+".mds")
	else:
		sys.stderr.write("[ERROR] Unkown Dimension reduction method\n")
		return None
	### process tsne
	tsne = TSNE(n_components=ntsne, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
	tsne_results = tsne.fit_transform(X)
	statplot.groups_scatter_flatdata(tsne_results[:,0],tsne_results[:,1],labels,"Dimension 1","Dimension 2",fig_prefix=plotprefix+"."+method+".tnse")
	return tsne_results

	
if __name__ == "__main__":
	"""
	print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
	print "Running example on 2,500 MNIST digits..."
	X = Math.loadtxt(sys.argv[1]);
	X = X.T
	mutilstats.centring(X)
	#mutilstats.normalize(X)
	labels = Math.loadtxt(sys.argv[2]);
	Y = tsne(X, 2, 50, 20);
	Plot.scatter(Y[:,0], Y[:,1], 20, labels,alpha=0.5);
	Plot.savefig("test.tsne.png",format='png',dpi=300)
	Plot.clf();Plot.close()
	"""
	##  process: partly_translated_3mer.txt transcribed_3mer.txt translated_3mer.txt
	fns = sys.argv[1:]
	classname = ["partly","transcribed","translated"]
	Y = []
	X = []
	for i in range(len(fns)):
		f = open(fns[i],"r")
		for line in f:
			if line.startswith("#"):continue
			X.append(list(map(float,line.rstrip("\n").split("\t")[1:])))
			Y.append(classname[i])
		f.close()
	assert len(X) == len(Y)
	X = np.asarray(X)
	print(X.shape)
	print(len(Y))
	tsne2(X,Y,ntsne=2,ndim=10,method="mds",plotprefix="plot",verbose=1,perplexity=40,n_iter=300)	
