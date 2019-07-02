# -*- coding: UTF-8 -*-
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import cycler
import itertools

#print cm._cmapnames
#['Spectral', 'copper', 'RdYlGn', 'Set2', 'summer', 'spring', 'gist_ncar', 'terrain', 'OrRd', 'RdBu', 'autumn', 'gist_earth', 'Set1', 'PuBu', 'Set3', 'brg', 'gnuplot2', 'gist_rainbow', 'pink', 'binary', 'winter', 'jet', 'BuPu', 'Dark2', 'prism', 'Oranges', 'gist_yarg', 'BuGn', 'hot', 'PiYG', 'YlOrBr', 'PRGn', 'Reds', 'spectral', 'bwr', 'RdPu', 'cubehelix', 'Greens', 'rainbow', 'Accent', 'gist_heat', 'YlGnBu', 'RdYlBu', 'Paired', 'flag', 'hsv', 'BrBG', 'seismic', 'Blues', 'Purples', 'cool', 'Pastel2', 'gray', 'coolwarm', 'Pastel1', 'gist_stern', 'gnuplot', 'GnBu', 'YlGn', 'Greys', 'RdGy', 'ocean', 'YlOrRd', 'PuOr', 'PuRd', 'gist_gray', 'CMRmap', 'PuBuGn', 'afmhot', 'bone']

# ggplot axes colors [u'#E24A33', u'#348ABD', u'#988ED5', u'#777777', u'#FBC15E', u'#8EBA42', u'#FFB5B8']

# for projection='3d'
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import fcluster; import pandas
mpl.rcParams['grid.alpha'] = 1.0
mpl.rcParams['pgf.texsystem'] = 'xelatex'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['axes.labelcolor'] = u'#000000'
mpl.rcParams['text.color'] = 'k'
mpl.rcParams['lines.dash_joinstyle'] = 'round'
mpl.rcParams['text.latex.unicode'] = False
mpl.rcParams['path.simplify_threshold'] = 0.11111111111
mpl.rcParams['ytick.labelsize'] = 'medium'
mpl.rcParams['ps.papersize'] = 'letter'
mpl.rcParams['grid.color'] = 'gray'
mpl.rcParams['axes.prop_cycle'] = cycler(u"color",[u'#E24A33', u'#348ABD', u'#988ED5', u'#777777', u'#FBC15E', u'#8EBA42', u'#FFB5B8'])
#mpl.rcParams['axes.facecolor'] = u"#FFFFF0" #axes.facecolor
mpl.rcParams['ytick.color'] = u'#000000'
mpl.rcParams['xtick.color'] = u'#000000'
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
#mpl.rcParams['ps.fonttype'] = 3
mpl.rcParams['path.simplify'] = True
mpl.rcParams['toolbar'] = 'toolbar2'
mpl.rcParams['axes.linewidth'] = .8
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['lines.markeredgewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['axes.labelcolor'] = u'#000000'
#mpl.rcParams['xtick.color'] = u'#000000'
#mpl.rcParams['ytick.color'] = u'#000000'
mpl.rcParams['axes.edgecolor'] = u'#000000'
mpl.rcParams['text.color'] = u'#000000'

mpl.rcParams['axes.grid'] = False
mpl.rcParams['font.sans-serif'] = [u'Arial',u'Helvetica',u'Bitstream Vera Sans', u'DejaVu Sans', u'Lucida Grande', u'Verdana', u'Geneva', u'Lucid', u'Arial',u'Helvetica', u'Avant Garde', u'sans-serif']
mpl.rcParams['font.serif'] = [u'Times New Roman', u'Palatino', u'New Century Schoolbook', u'Bookman', u'Computer Modern Roman'] 
mpl.rcParams['font.weight'] = 'normal'
#mpl.rcParams['font.monospace'] = [u'Bitstream Vera Sans Mono', u'DejaVu Sans Mono', u'Andale Mono', u'Nimbus Mono L', u'Courier New', u'Courier',u'Fixed', u'Terminal', u'monospace']
mpl.rcParams['text.usetex'] = 'false'
mpl.rcParams['savefig.edgecolor'] = 'w' 
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['font.stretch'] = 'normal'
mpl.rcParams['font.family'] = [u'sans-serif',u'serif']
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['mathtext.it'] = u'serif:italic'
mpl.rcParams['mathtext.bf'] = u'serif:bold'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['xtick.direction'] = "out"
mpl.rcParams['ytick.direction'] = "out"
mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['legend.scatterpoints'] = 1
###
from matplotlib.patches import Polygon
# to get kmeans and scipy.cluster.hierarchy
from scipy.cluster.vq import *
from scipy.cluster.hierarchy import *

###
from matplotlib.colors import LogNorm

##kmeans归一化处理 from scipy.cluster.vq import whiten
from scipy.cluster.vq import whiten

#color define 
colordefine = {
		##sci colors : https://cran.r-project.org/web/packages/ggsci/vignettes/ggsci.html
		##color data : https://github.com/nanxstats/ggsci/blob/master/data-raw/data-generator.R
		#Color palette inspired by plots in Nature Reviews Cancer
		"npg": ["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F","#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"],
		#Color palette inspired by plots in Science from AAAS
		"aaas":["#3B4992","#EE0000","#008B45","#631879","#008280","#BB0021","#5F559B","#A20056","#808180","#1B1919"],
		#Color palette inspired by plots in The New England Journal of Medicine
		"nejm":["#BC3C29","#0072B5","#E18727","#20854E","#7876B1","#6F99AD","#FFDC91","#EE4C97"],
		#Color palette inspired by plots in Lancet Oncology
		"lancet":["#00468B","#ED0000","#42B540","#0099B4","#925E9F","#FDAF91","#AD002A","#ADB6B6","#1B1919"],
		#Color palette inspired by plots in The Journal of the American Medical Association
		"jama":["#374E55","#DF8F44","#00A1D5","#B24745","#79AF97","#6A6599","#80796B"],
		#Color palette inspired by plots in Journal of Clinical Oncology
		"jco": ["#0073C2","#EFC000","#868686","#CD534C","#7AA6DC","#003C67","#8F7700","#3B3B3B","#A73030","#4A6990"],
		# Color palette inspired by D3.js category10
		"d3c10":["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"],
		# Color palette inspired by D3.js category20
		"d3c20":["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF","#AEC7E8","#FFBB78","#98DF8A","#FF9896","#C5B0D5","#C49C94","#F7B6D2","#C7C7C7","#DBDB8D","#9EDAE5"],
		# Color palette inspired by D3.js category20b
		"d3c20b":["#393B79","#637939","#8C6D31","#843C39","#7B4173","#5254A3","#8CA252","#BD9E39","#AD494A","#A55194","#A55194","#B5CF6B","#E7BA52","#D6616B","#CE6DBD","#9C9EDE","#CEDB9C","#E7CB94","#E7969C","#DE9ED6"],
		#Color palette inspired by D3.js category20c
		"d3c20c":["#3182BD","#E6550D","#31A354","#756BB1","#636363","#6BAED6","#FD8D3C","#74C476","#9E9AC8","#969696","#9ECAE1","#FDAE6B","#A1D99B","#BCBDDC","#BDBDBD","#C6DBEF","#FDD0A2","#C7E9C0","#DADAEB","#D9D9D9"],
		#Color palette inspired by LocusZoom
		"locuszoom":["#D43F3A","#EEA236","#5CB85C","#46B8DA","#357EBD","#9632B8","#B8B8B8"],
		#Color palette inspired by University of Chicago Color Palette
		"uchicago":["#800000","#767676","#FFA319","#8A9045","#155F83","#C16622","#8F3931","#58593F","#350E20"],
		#Color palette inspired by University of Chicago color palette (light version)
		"uchicagolight":["#800000","#D6D6CE","#FFB547","#ADB17D","#5B8FA8","#D49464","#B1746F","#8A8B79","#725663"],
		#Color palette inspired by University of Chicago color palette (dark version)
		"uchicagodark":["#800000","#767676","#CC8214","#616530","#0F425C","#9A5324","#642822","#3E3E23","#350E20"],
		#Color palette inspired by The Simpsons "simpsons"$"springfield"
		"simpsons":["#FED439","#709AE1","#8A9197","#D2AF81","#FD7446","#D5E4A2","#197EC0","#F05C3B","#46732E","#71D0F5","#370335","#075149","#C80813","#91331F","#1A9993","#FD8CC1"],
		# Color palette inspired by Futurama
		"futurama":["#FF6F00","#C71000","#008EA0","#8A4198","#5A9599","#FF6348","#84D7E1","#FF95A8","#3D3B25","#ADE2D0","#1A5354","#3F4041"],
		#Color palette inspired by Rick and Morty "rickandmorty"$"schwifty" 
		"rickandmorty":["#FAFD7C","#82491E","#24325F","#B7E4F9","#FB6467","#526E2D","#E762D7","#E89242","#FAE48B","#A6EEE6","#917C5D","#69C8EC"],
		#Color palette inspired by Star Trek "startrek"$"uniform"
		"startrek":["#CC0C00","#5C88DA","#84BD00","#FFCD00","#7C878E","#00B5E2","#00AF66"],
		# Color palette inspired by Tron Legacy
		"tron":["#FF410D","#6EE2FF","#F7C530","#95CC5E","#D0DFE6","#F79D1E","#748AA6"],
		##
		"bmh": [u'#348ABD', u'#A60628', u'#7A68A6', u'#467821', u'#D55E00', u'#CC79A7', u'#56B4E9', u'#009E73', u'#F0E442'],
		"ggplot1": [u'#E41A1C', u'#377EB8', u'#4DAF4A', u'#984EA3', u'#FF7F00', u'#FFFF33', u'#A65628', u'#F781BF', u'#999999'], 
		"ggplot2": ["#66C2A5", "#FC8D62","#8DA0CB","#E78AC3","#A6D854","#FFD92F","#E5C494","#B3B3B3"],
		"ggplot3": ["#FF6C91","#BC9D00","#00BB57","#00B8E5","#CD79FF"],
		"tableau10M": ["#609DCA","#FF9641","#38C25D","#FF5B4E","#B887C3","#B67365","#FE90C2","#A4A09B","#D2CC5A"],
		"tableau10": ["#0076AE","#FF7400","#00A13B","#EF0000","#9E63B5","#985247","#F66EB8","#7F7C77","#C2BD2C"],
		"seaborn": ["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974","#64B5CD"],
		"seabornHSUL": ["#F67088","#CE8F31","#96A331","#32B165", "#35ACA4", "#38A7D0", "#A38CF4","#F461DD"],
		"D3": ["#5E9CC6","#FF7D0B", "#2CA02C", "#D62728", "#9467BD","#8C564B"],
		"custom1":["#db4d4d","#9b75b5","#6471af","#24bcb8","#b23b9c","#eace54","#878787","#aa6357","#70aa7a"],
		"custom2":["#9b75b5","#878787","#6471af","#b23b9c","#aa6357","#70aa7a","#db4d4d","#24bcb8","#eace54"],
		}
colorCMs = "Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r".split(", ")

def __getallcolors():
	keys = []
	for key in colordefine.keys() + colorCMs:
		#if key.endswith("_r"):continue
		keys.append(key)
	return keys

def makestyles(a,n):
	assert len(a) > 0
	if isinstance(a,list):
		gen =  itertools.cycle(itertools.chain(a))
	elif isinstance(a,str):
		a = [a,]
		gen =  itertools.cycle(itertools.chain(a))
	else:
		raise
	ret = []
	for i in xrange(n):
		ret.append(gen.next())
	return ret

def styles(num,colorgrad=None,defaultnum = 8):
	if colorgrad is None:colorgrad = "npg"
	elif not isinstance(colorgrad,str):
		try:
			colorgrad = colorgrad.name
		except:
			sys.stderr.write("[ERROR] unkown colorgrad name\n")
			sys.exit(1)
	if colorgrad in colordefine:
		axescolors = colordefine[colorgrad]
	elif colorgrad in colorCMs:
		stylecolors = color_grad(max(num,defaultnum),colorgrad=colorgrad)
		axescolors = map(matplotlib.colors.rgb2hex,stylecolors)
	else:
		sys.stderr.write("[ERROR] unkown colorgrad name\n")
		sys.exit(1)
	color_raw = itertools.cycle(itertools.chain(axescolors))
	lines_raw = itertools.cycle(itertools.chain(['-','--','-.',':']))
	marker_raw = itertools.cycle(itertools.chain(['o','^','s','+','*','D','v','1','2','x','3','4','s','p','>','<','h','d','H']))
	
	ret_color = []
	ret_lines = []
	ret_marker = []
	for i in xrange(num):
		ret_color.append(color_raw.next())
		ret_lines.append(lines_raw.next())
		ret_marker.append(marker_raw.next())
	return ret_color,ret_lines,ret_marker

def inscolor(stringlist,colorgrad = "npg"):
	ret_color,ret_lines,ret_marker = styles(len(stringlist),colorgrad = colorgrad)
	h = {}
	for i in xrange(len(stringlist)):
		h[stringlist[i]] = ret_color[i]
	return h

def liststyle(datalist,colorgrad = "npg"):
	setdatalist = list(set(datalist))
	ret_color,ret_lines,ret_marker = styles(len(setdatalist),colorgrad = colorgrad)
	h = {}
	for i in xrange(len(setdatalist)):
		h[setdatalist[i]] = i
	colorlist,lineslist,markerlist = [],[],[]
	for i in xrange(len(datalist)):
		idx = h[datalist[i]]
		colorlist.append(ret_color[idx])
		lineslist.append(ret_lines[idx])
		markerlist.append(ret_marker[idx])
	return colorlist,lineslist,markerlist


## this is not same as sequential colormaps 
def color_grad(num,colorgrad="npg"):
	cminstance = cm.get_cmap(colorgrad,num)
	color_class = cminstance(np.linspace(0, 1, num))
	return color_class


def rgb2hex(rgbORrgba):
	return matplotlib.colors.rgb2hex(rgbORrgba)

if __name__ == "__main__":
	#print color_grad(3)
	colornames = __getallcolors()
	import numpy as np
	## plot hist norm
	from rblib import statplot
	data = []
	names = []
	for i in xrange(5):
		names.append("S%d"%(i+1))
		data.append(np.random.randn(1000) + i*2.2)
	statplot.hist_groups(data,names,"data","testplot_hist",alpha=0.8,normed=False,bins=10,rwidth=0.8,hist=True,figsize=(6,5),histtype="bar")



"""
113     #   plot(x, y, color='green', linestyle='dashed', marker='o',
114     #                    markerfacecolor='blue', markersize=12).
115     # 因此，产生3个元素列表，for  colorstyle， linestyle， makerstyle  ##
116     #1247 'b'         blue
117     #1248 'g'         green
118     #1249 'r'         red
119     #1250 'c'         cyan
120     #1251 'm'         magenta
121     #1252 'y'         yellow
122     #1253 'k'         black
123     #1271     ================    ===============================
124     #1272     character           description
125     #1273     ================    ===============================
126     #1274     ``'-'``             solid line style
127     #1275     ``'--'``            dashed line style
128     #1276     ``'-.'``            dash-dot line style
129     #1277     ``':'``             dotted line style
130     #1278     
131     #1279     
132     #1280     ``'.'``             point marker
133     #1281     ``','``             pixel marker
134     #1282     ``'o'``             circle marker
135     #1283     ``'v'``             triangle_down marker
136     #1284     ``'^'``             triangle_up marker
137     #1285     ``'<'``             triangle_left marker
138     #1286     ``'>'``             triangle_right marker
139     #1287     ``'1'``             tri_down marker
140     #1288     ``'2'``             tri_up marker
141     #1289     ``'3'``             tri_left marker
142     #1290     ``'4'``             tri_right marker
143     #1291     ``'s'``             square marker
144     #1292     ``'p'``             pentagon marker
145     #1293     ``'*'``             star marker
146     #1294     ``'h'``             hexagon1 marker
147     #1295     ``'H'``             hexagon2 marker
148     #1296     ``'+'``             plus marker
149     #1297     ``'x'``             x marker
150     #1298     ``'D'``             diamond marker
151     #1299     ``'d'``             thin_diamond marker
152     #1300     ``'|'``             vline marker
153     #1301     ``'_'``             hline marker
154     #1302 
155     #1303 marker: [ ``7`` | ``4`` | ``5`` | ``6`` | ``'o'`` | ``'D'`` | ``'h'`` | ``'H'`` | ``'_'`` | ``''`` | ``'None'`` | ``' '`` | ``None`` |        ``'8'`` | ``'p'``      | ``','`` | ``'+'`` | ``'.'`` | ``'s'`` | ``'*'`` | ``'d'`` | ``3`` | ``0`` | ``1`` | ``2`` | ``'1'`` | ``'3'`` | ``'4'`` |      ``'2'`` | ``'v'`` | ``'<'``       | ``'>'`` | ``'^'`` | ``'|'`` | ``'x'`` | ``'$...$'`` | *tuple* | *Nx2 array* ]
"""


"""

# Color palette inspired by UCSC Genome Browser Chromosome Colors
ggsci_db$"ucscgb"$"default" <- c(
  "chr5" = "#FF0000", "chr8" = "#FF9900", "chr9" = "#FFCC00",
  "chr12" = "#00FF00", "chr15" = "#6699FF", "chr20" = "#CC33FF",
  "chr3" = "#99991E", "chrX" = "#999999", "chr6" = "#FF00CC",
  "chr4" = "#CC0000", "chr7" = "#FFCCCC", "chr10" = "#FFFF00",
  "chr11" = "#CCFF00", "chr13" = "#358000", "chr14" = "#0000CC",
  "chr16" = "#99CCFF", "chr17" = "#00FFFF", "chr18" = "#CCFFFF",
  "chr19" = "#9900CC", "chr21" = "#CC99FF", "chr1" = "#996600",
  "chr2" = "#666600", "chr22" = "#666666", "chrY" = "#CCCCCC",
  "chrUn" = "#79CC3D", "chrM" = "#CCCC99"
)

# Color palette inspired by IGV
ggsci_db$"igv"$"default" <- c(
  "chr1" = "#5050FF", "chr2" = "#CE3D32", "chr3" = "#749B58",
  "chr4" = "#F0E685", "chr5" = "#466983", "chr6" = "#BA6338",
  "chr7" = "#5DB1DD", "chr8" = "#802268", "chr9" = "#6BD76B",
  "chr10" = "#D595A7", "chr11" = "#924822", "chr12" = "#837B8D",
  "chr13" = "#C75127", "chr14" = "#D58F5C", "chr15" = "#7A65A5",
  "chr16" = "#E4AF69", "chr17" = "#3B1B53", "chr18" = "#CDDEB7",
  "chr19" = "#612A79", "chr20" = "#AE1F63", "chr21" = "#E7C76F",
  "chr22" = "#5A655E", "chrX" = "#CC9900", "chrY" = "#99CC00",
  "chrUn" = "#A9A9A9", "chr23" = "#CC9900", "chr24" = "#99CC00",
  "chr25" = "#33CC00", "chr26" = "#00CC33", "chr27" = "#00CC99",
  "chr28" = "#0099CC", "chr29" = "#0A47FF", "chr30" = "#4775FF",
  "chr31" = "#FFC20A", "chr32" = "#FFD147", "chr33" = "#990033",
  "chr34" = "#991A00", "chr35" = "#996600", "chr36" = "#809900",
  "chr37" = "#339900", "chr38" = "#00991A", "chr39" = "#009966",
  "chr40" = "#008099", "chr41" = "#003399", "chr42" = "#1A0099",
  "chr43" = "#660099", "chr44" = "#990080", "chr45" = "#D60047",
  "chr46" = "#FF1463", "chr47" = "#00D68F", "chr48" = "#14FFB1"
)

# Color palette inspired by IGV
ggsci_db$"igv"$"alternating" <- c(
  "Indigo" = "#5773CC", "SelectiveYellow" = "#FFB900"
)

# Color palette inspired by COSMIC Hallmarks of Cancer
ggsci_db$"cosmic"$"hallmarks_dark" <- c(
  "Invasion and Metastasis" = "#171717",
  "Escaping Immunic Response to Cancer" = "#7D0226",
  "Change of Cellular Energetics" = "#300049",
  "Cell Replicative Immortality" = "#165459",
  "Suppression of Growth" = "#3F2327",
  "Genome Instability and Mutations" = "#0B1948",
  "Angiogenesis" = "#E71012",
  "Escaping Programmed Cell Death" = "#555555",
  "Proliferative Signaling" = "#193006",
  "Tumour Promoting Inflammation" = "#A8450C"
)

# Color palette inspired by Hanahan, Weinberg Hallmarks of Cancer
ggsci_db$"cosmic"$"hallmarks_light" <- c(
  "Invasion and Metastasis" = "#2E2A2B",
  "Escaping Immunic Response to Cancer" = "#CF4E9C",
  "Change of Cellular Energetics" = "#8C57A2",
  "Cell Replicative Immortality" = "#358DB9",
  "Suppression of Growth" = "#82581F",
  "Genome Instability and Mutations" = "#2F509E",
  "Angiogenesis" = "#E5614C",
  "Escaping Programmed Cell Death" = "#97A1A7",
  "Proliferative Signaling" = "#3DA873",
  "Tumour Promoting Inflammation" = "#DC9445"
)

# Color palette inspired by COSMIC Hallmarks of Cancer
ggsci_db$"cosmic"$"signature_substitutions" <- c(
  "C>A" = "#5ABCEB",
  "C>G" = "#050708",
  "C>T" = "#D33C32",
  "T>A" = "#CBCACB",
  "T>C" = "#ABCD72",
  "T>G" = "#E7C9C6"
)


# Color palette inspired by heatmaps generated by GSEA GenePattern
ggsci_db$"gsea"$"default" <- c(
  "Purple" = "#4500AD", "DarkBlue" = "#2700D1",
  "RoyalBlue" = "#6B58EF", "Malibu" = "#8888FF",
  "Melrose" = "#C7C1FF", "Fog" = "#D5D5FF",
  "CottonCandy" = "#FFC0E5", "VividTangerine" = "#FF8989",
  "BrinkPink" = "#FF7080", "Persimmon" = "#FF5A5A",
  "Flamingo" = "#EF4040", "GuardsmanRed" = "#D60C00"
)

# Material Design color palettes
ggsci_db$"material"$"red" <- c(
  "Red50" = "#FFEBEE", "Red100" = "#FFCDD2",
  "Red200" = "#EF9A9A", "Red300" = "#E57373",
  "Red400" = "#EF5350", "Red500" = "#F44336",
  "Red600" = "#E53935", "Red700" = "#D32F2F",
  "Red800" = "#C62828", "Red900" = "#B71C1C"
)

ggsci_db$"material"$"pink" <- c(
  "Pink50" = "#FCE4EC", "Pink100" = "#F8BBD0",
  "Pink200" = "#F48FB1", "Pink300" = "#F06292",
  "Pink400" = "#EC407A", "Pink500" = "#E91E63",
  "Pink600" = "#D81B60", "Pink700" = "#C2185B",
  "Pink800" = "#AD1457", "Pink900" = "#880E4F"
)

ggsci_db$"material"$"purple" <- c(
  "Purple50" = "#F3E5F5", "Purple100" = "#E1BEE7",
  "Purple200" = "#CE93D8", "Purple300" = "#BA68C8",
  "Purple400" = "#AB47BC", "Purple500" = "#9C27B0",
  "Purple600" = "#8E24AA", "Purple700" = "#7B1FA2",
  "Purple800" = "#6A1B9A", "Purple900" = "#4A148C"
)

ggsci_db$"material"$"deep-purple" <- c(
  "DeepPurple50" = "#EDE7F6", "DeepPurple100" = "#D1C4E9",
  "DeepPurple200" = "#B39DDB", "DeepPurple300" = "#9575CD",
  "DeepPurple400" = "#7E57C2", "DeepPurple500" = "#673AB7",
  "DeepPurple600" = "#5E35B1", "DeepPurple700" = "#512DA8",
  "DeepPurple800" = "#4527A0", "DeepPurple900" = "#311B92"
)

ggsci_db$"material"$"indigo" <- c(
  "Indigo50" = "#E8EAF6", "Indigo100" = "#C5CAE9",
  "Indigo200" = "#9FA8DA", "Indigo300" = "#7986CB",
  "Indigo400" = "#5C6BC0", "Indigo500" = "#3F51B5",
  "Indigo600" = "#3949AB", "Indigo700" = "#303F9F",
  "Indigo800" = "#283593", "Indigo900" = "#1A237E"
)

ggsci_db$"material"$"blue" <- c(
  "Blue50" = "#E3F2FD", "Blue100" = "#BBDEFB",
  "Blue200" = "#90CAF9", "Blue300" = "#64B5F6",
  "Blue400" = "#42A5F5", "Blue500" = "#2196F3",
  "Blue600" = "#1E88E5", "Blue700" = "#1976D2",
  "Blue800" = "#1565C0", "Blue900" = "#0D47A1"
)

ggsci_db$"material"$"light-blue" <- c(
  "LightBlue50" = "#E1F5FE", "LightBlue100" = "#B3E5FC",
  "LightBlue200" = "#81D4FA", "LightBlue300" = "#4FC3F7",
  "LightBlue400" = "#29B6F6", "LightBlue500" = "#03A9F4",
  "LightBlue600" = "#039BE5", "LightBlue700" = "#0288D1",
  "LightBlue800" = "#0277BD", "LightBlue900" = "#01579B"
)

ggsci_db$"material"$"cyan" <- c(
  "Cyan50" = "#E0F7FA", "Cyan100" = "#B2EBF2",
  "Cyan200" = "#80DEEA", "Cyan300" = "#4DD0E1",
  "Cyan400" = "#26C6DA", "Cyan500" = "#00BCD4",
  "Cyan600" = "#00ACC1", "Cyan700" = "#0097A7",
  "Cyan800" = "#00838F", "Cyan900" = "#006064"
)

ggsci_db$"material"$"teal" <- c(
  "Teal50" = "#E0F2F1", "Teal100" = "#B2DFDB",
  "Teal200" = "#80CBC4", "Teal300" = "#4DB6AC",
  "Teal400" = "#26A69A", "Teal500" = "#009688",
  "Teal600" = "#00897B", "Teal700" = "#00796B",
  "Teal800" = "#00695C", "Teal900" = "#004D40"
)

ggsci_db$"material"$"green" <- c(
  "Green50" = "#E8F5E9", "Green100" = "#C8E6C9",
  "Green200" = "#A5D6A7", "Green300" = "#81C784",
  "Green400" = "#66BB6A", "Green500" = "#4CAF50",
  "Green600" = "#43A047", "Green700" = "#388E3C",
  "Green800" = "#2E7D32", "Green900" = "#1B5E20"
)

ggsci_db$"material"$"light-green" <- c(
  "LightGreen50" = "#F1F8E9", "LightGreen100" = "#DCEDC8",
  "LightGreen200" = "#C5E1A5", "LightGreen300" = "#AED581",
  "LightGreen400" = "#9CCC65", "LightGreen500" = "#8BC34A",
  "LightGreen600" = "#7CB342", "LightGreen700" = "#689F38",
  "LightGreen800" = "#558B2F", "LightGreen900" = "#33691E"
)

ggsci_db$"material"$"lime" <- c(
  "Lime50" = "#F9FBE7", "Lime100" = "#F0F4C3",
  "Lime200" = "#E6EE9C", "Lime300" = "#DCE775",
  "Lime400" = "#D4E157", "Lime500" = "#CDDC39",
  "Lime600" = "#C0CA33", "Lime700" = "#AFB42B",
  "Lime800" = "#9E9D24", "Lime900" = "#827717"
)

ggsci_db$"material"$"yellow" <- c(
  "Yellow50" = "#FFFDE7", "Yellow100" = "#FFF9C4",
  "Yellow200" = "#FFF59D", "Yellow300" = "#FFF176",
  "Yellow400" = "#FFEE58", "Yellow500" = "#FFEB3B",
  "Yellow600" = "#FDD835", "Yellow700" = "#FBC02D",
  "Yellow800" = "#F9A825", "Yellow900" = "#F57F17"
)

ggsci_db$"material"$"amber" <- c(
  "Amber50" = "#FFF8E1", "Amber100" = "#FFECB3",
  "Amber200" = "#FFE082", "Amber300" = "#FFD54F",
  "Amber400" = "#FFCA28", "Amber500" = "#FFC107",
  "Amber600" = "#FFB300", "Amber700" = "#FFA000",
  "Amber800" = "#FF8F00", "Amber900" = "#FF6F00"
)

ggsci_db$"material"$"orange" <- c(
  "Orange50" = "#FFF3E0", "Orange100" = "#FFE0B2",
  "Orange200" = "#FFCC80", "Orange300" = "#FFB74D",
  "Orange400" = "#FFA726", "Orange500" = "#FF9800",
  "Orange600" = "#FB8C00", "Orange700" = "#F57C00",
  "Orange800" = "#EF6C00", "Orange900" = "#E65100"
)

ggsci_db$"material"$"deep-orange" <- c(
  "DeepOrange50" = "#FBE9E7", "DeepOrange100" = "#FFCCBC",
  "DeepOrange200" = "#FFAB91", "DeepOrange300" = "#FF8A65",
  "DeepOrange400" = "#FF7043", "DeepOrange500" = "#FF5722",
  "DeepOrange600" = "#F4511E", "DeepOrange700" = "#E64A19",
  "DeepOrange800" = "#D84315", "DeepOrange900" = "#BF360C"
)

ggsci_db$"material"$"brown" <- c(
  "Brown50" = "#EFEBE9", "Brown100" = "#D7CCC8",
  "Brown200" = "#BCAAA4", "Brown300" = "#A1887F",
  "Brown400" = "#8D6E63", "Brown500" = "#795548",
  "Brown600" = "#6D4C41", "Brown700" = "#5D4037",
  "Brown800" = "#4E342E", "Brown900" = "#3E2723"
)

ggsci_db$"material"$"grey" <- c(
  "Grey50" = "#FAFAFA", "Grey100" = "#F5F5F5",
  "Grey200" = "#EEEEEE", "Grey300" = "#E0E0E0",
  "Grey400" = "#BDBDBD", "Grey500" = "#9E9E9E",
  "Grey600" = "#757575", "Grey700" = "#616161",
  "Grey800" = "#424242", "Grey900" = "#212121"
)

ggsci_db$"material"$"blue-grey" <- c(
  "BlueGrey50" = "#ECEFF1", "BlueGrey100" = "#CFD8DC",
  "BlueGrey200" = "#B0BEC5", "BlueGrey300" = "#90A4AE",
  "BlueGrey400" = "#78909C", "BlueGrey500" = "#607D8B",
  "BlueGrey600" = "#546E7A", "BlueGrey700" = "#455A64",
  "BlueGrey800" = "#37474F", "BlueGrey900" = "#263238"
)


"""




