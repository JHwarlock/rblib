# This is used for plot network and do network analysis
##
import sys
try:
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	matplotlib.style.use('ggplot')
	import numpy as np
except:
	raise

# first to define the plot attr
class Style(object):
	def __init__(self):
		self.nodeshape = list("so^>v<dph8")
		self.nodecolor = 'r'
		self.nodesize = 300
		self.node_another = None
		self.edgeshape = ["solid","dashed","dotted","dashdot"]
		self.edgecolor = 'b'
		self.edgesize  = 1 # width
		self.linewidths = 0  # node line width
def create_style():
	style = Style()
	return style

# 0. node name
# 1. node shape # so^>v<dph8  
# 2. node color  ## use float or int
# 3. node size   ## use int (default=300) , now is define by degree
# 4. node edge color  (note define now)
# 5. alpha (will update)
# 
#  edges 
# 0. edge1
# 1. edge2
# 2. edges shape ##  Edge line style (solid | dashed | dotted | dashdot)
# 3. edges color (not define now)
# 4. edges size (not define now) 
# 5. alpha (will update)

#  test size 

import networkx as nx

def node_parse(fn):
	f = file(fn,"r")
	hnodeattr = {}
	vmax = 0
	for line in f:
		if line.startswith("#"):continue
		nodename,nodeshape,nodecolor,nodesize,nodeedgecolor = line.rstrip("\n").split("\t")
		hnodeattr[nodename] = [int(nodeshape),float(nodecolor),float(nodesize),float(nodeedgecolor)]
		vmax = max(np.abs(float(nodecolor)),vmax)
	f.close()
	return hnodeattr,-1*vmax,vmax

def edge_parse(fn):
	f = file(fn,"r")
	hedgeattr = {}
	for line in f:
		if line.startswith("#"):continue
		edge1,edge2,edgeshape,edgecolor,edgesize = line.rstrip().split("\t")
		hedgeattr[edge1+"|||"+edge2] = [int(edgeshape),float(edgecolor),float(edgesize)]
		hedgeattr[edge2+"|||"+edge1] = [int(edgeshape),float(edgecolor),float(edgesize)]
	f.close()
	return hedgeattr

## 
# to define the grade color , some times use degree method 
# 

def build_ud_graph(edgeattr,nodeattr):
	hnode,nodemin,nodemax = node_parse(nodeattr)
	hedge = edge_parse(edgeattr)

	# build nx
	G = nx.Graph()
	# add node 
	for key in hnode:
		G.add_node(key,shape=hnode[key][0])
	# add edge
	for key in hedge:
		node1,node2 = key.split("|||")
		G.add_edge(node1,node2,shape=hedge[key][0],weight=1.0,length=5)

	return G,hnode,hedge,nodemin,nodemax

def pos_transpos(pos,scale=4):
	postrans = {}
	for node in pos:
		pos1,pos2 = pos[node]
		postrans[node] = (pos1*scale,pos2*scale)
	return postrans

def plot_ud_graph(edgeattr,nodeattr):
	plt.figure(figsize=(20,16))
	G,hnode,hedge,nodemin,nodemax = build_ud_graph(edgeattr,nodeattr)
	pos=nx.pygraphviz_layout(G,prog='neato')
	#pos = pos_transpos(pos)
	##pos = nx.spring_layout(G)
	#pos = nx.spectral_layout(G)
	tmpstyle = create_style()
	
	#print tmpstyle.nodeshape
	#node_color=[float(G.degree(v)) for v in G]
	node_size = [(300+100*float(G.degree(v))) for v in G]
	#node_degree = [float(G.degree(v)) for v in G]
	#nodemax = np.max(node_degree)
	#nodemin = np.min(node_degree)
	#node_shape = [ tmpstyle.nodeshape[hnode[v][0]]  for v in G]

	# nodes
	for i in xrange(len(tmpstyle.nodeshape)):
		nodelist = []
		#print G.nodes(data=True)
		for node,nodeattr in G.nodes(data=True):
			if nodeattr['shape'] == i:
				nodelist.append(node)
		node_color=[ hnode[v][1] for v in nodelist]
		node_size = [(100+30*float(G.degree(v))) for v in nodelist]
		nx.draw_networkx_nodes(G,pos,nodelist = nodelist,node_shape = tmpstyle.nodeshape[i],node_color=node_color,node_size=node_size,alpha=0.6,cmap = cm.cool,linewidths=0.1,vmin=nodemin,vmax=nodemax)
	# add_colorbar
	"""
	cmap=cm.cool
	sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=nodemin, vmax=nodemax))
	sm._A = []
	cb = plt.colorbar(sm,format="%.1f",shrink=0.5)
	cb.set_label('Differential Expression')
	cb.ax.yaxis.set_label_position('left')
	"""
	# edges
	for i in xrange(len(tmpstyle.edgeshape)):
		edgelist = []
		#print G.edges(data=True)
		for node1,node2,edgeattr in G.edges(data=True):
			if edgeattr['shape'] == i:
				edgelist.append((node1,node2))
		nx.draw_networkx_edges(G,pos,edgelist=edgelist,alpha=0.5,edge_color='gray',style=tmpstyle.edgeshape[i],width=1)
	nx.draw_networkx_labels(G,pos,font_size=5,font_family='sans-serif')

	plt.axis('off')
	plt.savefig("network.png",format='png',dpi=300)
	plt.savefig("network.svg",format='svg',dpi=300)
	plt.clf()
	plt.close()
	return 0

if __name__ == "__main__":
	edgeattr = sys.argv[1]
	nodeattr = sys.argv[2]
	plot_ud_graph(edgeattr,nodeattr)

"""
circular_layout(G[, dim, scale]) Position nodes on a circle.
random_layout(G[, dim]) Position nodes uniformly at random in the unit square.
shell_layout(G[, nlist, dim, scale]) Position nodes in concentric circles.
spring_layout(G[, dim, k, pos, fixed, ...]) Position nodes using Fruchterman-Reingold force-directed algorithm.
spectral_layout(G[, dim, weight, scale]) Position nodes using the eigenvectors of the graph Laplacian.
"""
