import sys
import numpy as np
from Bio import Phylo
from Bio.Phylo import TreeConstruction


def subdistmatrix(subnames,allnames,distmat):
	idx = []
	for num,name in enumerate(allnames):
		if name in subnames:
			idx.append(num)
	subdistmat = distmat[idx][:,idx]
	return subdistmat

def matrixformat(matrix):
	data = []
	n,p = matrix.shape
	assert n == p
	for i in range(n):
		data.append(matrix[i,0:i+1].tolist())
	return data

def dist2nw(names = ['Alpha', 'Beta', 'Gamma', 'Delta'],distmatrix = [[0], [1, 0], [2, 3, 0], [4, 5, 6, 0]],method="nj",fn = "test_evol.newick"):
	assert method in ["nj","upgma"]
	#distmatrix = matrixformat(distmatrix)
	dm = TreeConstruction._DistanceMatrix(names=names, matrix=distmatrix)
	constructor = TreeConstruction.DistanceTreeConstructor()
	# use UPGMA (Unweighted Pair Group Method with Arithmetic Mean) and NJ (Neighbor Joining). upgma
	if method == "nj": # Tree(rooted=False)
		tree = constructor.nj(dm)
	elif method == "upgma": # Tree(rooted=True)
		tree = constructor.upgma(dm)
	#Phylo.draw_ascii(tree)
	#print(tree)
	Phylo.write([tree,],fn,"newick")
	"""
	newick
	nexus
	nexml
	phyloxml
	cdao
	"""
	return 0

from ete3 import Tree, TreeStyle, NodeStyle, PhyloTree, CircleFace, TextFace, RectFace, random_color

def nw2dist(fn="test_evol.newick",treeflag=0,format=1):
	c = open(fn).read().strip()
	t = Tree(c,format)
	if treeflag: return t
	else:
		nodes = [n.name for n in t]
		n_nodes = len(nodes)
		distmat = np.zeros((n_nodes,n_nodes))
		for i in range(n_nodes-1):
			ni = nodes[i]
			for j in range(i+1,n_nodes):
				nj = nodes[j]
				d = t.get_distance(ni,nj)
				distmat[j,i] = distmat[i,j] = d
		return nodes,distmat
	# ancestor = t.get_common_ancestor("E","D")
	# t.set_outgroup(ancestor)
	# t.set_outgroup("B")

def	dist2file(names,distmat,fn):
	fo = open(fn,"w")
	fo.write("#\t%s\n"%("\t".join(names)))
	for i in range(len(names)):
		fo.write("%s\t%s\n"%(names[i],"\t".join(list(map(str,distmat[i].tolist())))))
	fo.close()
	return 0

if __name__ == "__main__":
	dist2nw()
	nodes,mat =  nw2dist()
	dist2file(nodes,mat,"out.evol.dist")

