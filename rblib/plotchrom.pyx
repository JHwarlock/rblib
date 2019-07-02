import sys
import os
import matplotlib
from rblib import statplot
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np

Karyocolors = {
		'gpos100' : (0/255.0,0/255.0,0/255.0),
		'gpos' : (0/255.0,0/255.0,0/255.0),
		'gpos75' : (130/255.0,130/255.0,130/255.0),
		'gpos66' : (160/255.0,160/255.0,160/255.0),
		'gpos50' : (200/255.0,200/255.0,200/255.0),
		'gpos33' : (210/255.0,210/255.0,210/255.0),
		'gpos25' : (200/255.0,200/255.0,200/255.0),
		'gvar' : (220/255.0,220/255.0,220/255.0),
		'gneg' : (255/255.0,255/255.0,255/255.0),
		'acen' : (217/255.0,47/255.0,39/255.0),
		'stalk' : (100/255.0,127/255.0,164/255.0),
		}


def parseregion(x,y,color,region):
	ret = []
	for i in xrange(len(region)-1):
		start = region[i]
		end = region[i+1]
		if x <= end and y >= start:# overlap
			ret.append([max(x,start),min(y,end),color])
		elif start > y:break
	return ret

def parsekaryo(karyo_filename):
	f = file(karyo_filename,"r")
	target = ["chr"+str(x) for x in range(1,23) + ["X","Y"]] # ,"M","MT"]]
	karyo_dict = {}
	center_dict = {}
	colornames = []
	for line in f:
		chrom,start,end,pq,colorname = line.rstrip("\n").split("\t")
		colornames.append(colorname)
		if not chrom.startswith("chr"): 
			chrom = "chr" + chrom
		if chrom in target:
			if chrom not in karyo_dict:karyo_dict[chrom] = []
			if colorname == "acen":
				if chrom not in center_dict:
					center_dict[chrom] = []
				center_dict[chrom].extend([float(start),float(end)])
			karyo_dict[chrom].append([chrom,float(start),float(end),pq,colorname])
	f.close()
	for chrom in center_dict:
		karyo_dict[chrom+"_center"] = (min(center_dict[chrom]) + max(center_dict[chrom]))/2
	return karyo_dict,list(set(colornames))

class Plotchrom(object):
	def __init__(self,karyo_dict,unit=500000,ax = None,fig = None,height=5,ystart=0,orders=["chr3",]):
		figwidth = 20
		figheigt = 2
		self.fig = plt.figure(figsize=(figwidth,figheigt),dpi=300) if fig == None else fig
		self.ax =  self.fig.add_subplot(111) if ax == None else ax
		self.unit = float(unit)
		self.height = height
		self.karyo_dict = karyo_dict
		self.ystart = ystart # record the first chrom 's position on yaxis
		self.plotorders = {}
		for i in xrange(len(orders)):
			self.plotorders[orders[i]] = i 

	def __get_chrom_len(self,chromosome="chr1"): # chromosome: "chr1"
		chromosome_start = float(min([x[1] for x in self.karyo_dict[chromosome]]))
		chromosome_end = float(max(x[2] for x in self.karyo_dict[chromosome]))
		chromosome_length = chromosome_end - chromosome_start
		centromere = self.karyo_dict[chromosome + "_center"] if chromosome + "_center" in self.karyo_dict else None
		return chromosome_length,centromere

	def save(self,figprefix,adjust=1,fmt="both"):
		statplot.clean_axis(self.ax)
		if adjust: self.fig.tight_layout()
		if fmt == "both":
			plt.savefig(figprefix+".png",format='png',dpi=300)
			plt.savefig(figprefix+".svg",format='svg',dpi=300)
		else:
			plt.savefig(figprefix+"."+fmt,format=fmt,dpi=300)
		sys.stderr.write("[INFO] png and svg has been saved with prefix '%s'\n"%figprefix)
		plt.clf();plt.close();
		return 0

	def __get_semi_y(self,x,x0,y0,r,coff=1):
		tmp = r**2-(x-x0)**2
		tmp[tmp<0] = 0
		y = np.sqrt(tmp)*coff + y0
		return y

	def __plothalf(self,semic_radius,x1,x2,x0,y0,coff=1,bins=40,color="red",alpha=1.0):
		tmpx = np.linspace(x1,x2,bins)
		tmpy = self.__get_semi_y(tmpx,x0,y0,semic_radius,coff=coff)
		tmpx_new = np.zeros(len(tmpx) + 2); tmpx_new[0] = x1; tmpx_new[-1] = x2; tmpx_new[1:-1] = tmpx[:]
		tmpy_new = np.zeros(len(tmpx) + 2); tmpy_new[1:-1] = tmpy[:]; tmpy_new[0] = y0; tmpy_new[-1] = y0
		self.ax.fill(tmpx_new,tmpy_new,ec=color,fc=color,linewidth=0,alpha=alpha)
		return 0
	def __plotwhole(self,semic_radius,x1,x2,x0,y0,coff=1,bins=40,color="red",alpha=1.0):
		tmpx = np.linspace(x1,x2,bins)
		tmpx_new = np.zeros(len(tmpx) + 2); tmpx_new[0] = x1; tmpx_new[-1] = x2; tmpx_new[1:-1] = tmpx[:]
		tmpy = self.__get_semi_y(tmpx,x0,y0,semic_radius,coff=coff)
		tmpy1_new = np.zeros(len(tmpx) + 2); tmpy1_new[1:-1] = tmpy[:]; tmpy1_new[0] = y0; tmpy1_new[-1] = y0
		tmpy = self.__get_semi_y(tmpx,x0,y0,semic_radius,coff=coff*-1.0)
		tmpy2_new = np.zeros(len(tmpx) + 2); tmpy2_new[1:-1] = tmpy[:]; tmpy2_new[0] = y0; tmpy2_new[-1] = y0
		tmpx = np.concatenate((tmpx_new,tmpx_new))
		tmpy = np.concatenate((tmpy1_new,tmpy2_new))
		self.ax.fill(tmpx,tmpy,ec=color,fc=color,linewidth=0,alpha=alpha)
		return 0
		
	def plot_karyo(self,chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius):
		assert chrom in self.karyo_dict
		k = 0;
		for chrom,start,end,pq,color in self.karyo_dict[chrom]:
			tmpcolor = Karyocolors[color]
			startband = (start+1) / self.unit 
			endband   = (end+1)   / self.unit
			center_x1 = circle_end1; center_x2 = circle_start2; center_y1 = ystart + semic_radius
			if color == "acen":
				tmppos = [startband,endband][k]
				k += 1
				self.ax.fill([startband,tmppos,endband],[ystart + semic_radius, ystart + semic_radius*2 , ystart+semic_radius],ec=tmpcolor,fc=tmpcolor,linewidth=0)
				self.ax.fill([startband,tmppos,endband],[ystart + semic_radius, ystart , ystart+semic_radius],ec=tmpcolor,fc=tmpcolor,linewidth=0)
			elif circle_end1 <= startband <= endband<=circle_start2:
				r = Rectangle((startband,ystart), endband - startband , semic_radius*2 ,fc =tmpcolor,ec=tmpcolor,linewidth=0)
				self.ax.add_patch(r)
			# process for semi
			elif endband <=  circle_end1:
				self.__plotwhole(semic_radius,startband,endband,center_x1,center_y1,coff=1,color=tmpcolor)
			elif startband >= circle_start2:
				self.__plotwhole(semic_radius,startband,endband,center_x2,center_y1,coff=1,color=tmpcolor)
			elif startband < circle_end1 and endband > circle_end1:
				self.__plotwhole(semic_radius,startband,circle_end1,center_x1,center_y1,coff=1,color=tmpcolor)
				r = Rectangle((circle_end1,ystart), endband - circle_end1 , semic_radius*2 ,fc =tmpcolor,ec=tmpcolor,linewidth=0)
				self.ax.add_patch(r)
			elif endband > circle_start2 and startband < circle_start2:
				self.__plotwhole(semic_radius,circle_start2,endband,center_x2,center_y1,coff=1,color=tmpcolor)
				r = Rectangle((startband,ystart), circle_start2 - startband , semic_radius*2 ,fc =tmpcolor,ec=tmpcolor,linewidth=0)
				self.ax.add_patch(r)
		return 0
	## parseregion(x,y,color,region)
	def plot_karyo2(self,chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,centromere,alpha=1.0):
		assert chrom in self.karyo_dict
		center_x1 = circle_end1; center_x2 = circle_start2; center_y1 = ystart + semic_radius; cleft = centromere-semic_radius;cright=centromere + semic_radius;
		region = [circle_start1,circle_end1,cleft,centromere,cright,circle_start2,circle_end2]; rets = []
		for chrom,start,end,pq,color in self.karyo_dict[chrom]:
			startband = (start+1) / self.unit; endband   = (end+1)   / self.unit
			ret = parseregion(startband,endband,color,region)
			rets.extend(ret)
		for startband,endband,color in rets:
			tmpcolor = Karyocolors[color]
			if circle_end1 <= startband <= endband <= cleft or cright <= startband <= endband <= circle_start2:
				r = Rectangle((startband,ystart), endband - startband , semic_radius*2 ,fc =tmpcolor,ec=tmpcolor,linewidth=0,alpha=alpha)
				self.ax.add_patch(r)
			elif circle_start1 <= startband <= endband <=  circle_end1:
				self.__plotwhole(semic_radius,startband,endband,center_x1,center_y1,coff=1,color=tmpcolor,alpha=alpha)
			elif cleft <= startband <= endband <= centromere:
				self.__plotwhole(semic_radius,startband,endband,cleft,center_y1,coff=1,color=tmpcolor,alpha=alpha)
			elif centromere <= startband <= endband <= cright:
				self.__plotwhole(semic_radius,startband,endband,cright,center_y1,coff=1,color=tmpcolor,alpha=alpha)
			elif circle_start2 <= startband <= endband <= circle_end2:
				self.__plotwhole(semic_radius,startband,endband,center_x2,center_y1,coff=1,color=tmpcolor,alpha=alpha)
			else:
				sys.stderr.write("[WARN] maybe parse error for overlap region!\n")
		return 0
	def plot_halfcolorbin2(self,datalikekaryo,colordefine,chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,centromere,coff=1,alpha=1.0):
		center_x1 = circle_end1; center_x2 = circle_start2; center_y1 = ystart + semic_radius; cleft = centromere-semic_radius;cright=centromere + semic_radius;
		region = [circle_start1,circle_end1,cleft,centromere,cright,circle_start2,circle_end2]; rets = []
		for chrom,start,end,pq,color in datalikekaryo[chrom]:
			startband = (start+1) / self.unit; endband   = (end+1)   / self.unit
			ret = parseregion(startband,endband,color,region)
			rets.extend(ret)
		for startband,endband,color in rets:
			tmpcolor = colordefine[color]
			if circle_end1 <= startband <= endband <= cleft or cright <= startband <= endband <= circle_start2:
				r = Rectangle((startband,ystart+0.5*(1+coff)*semic_radius), endband - startband , semic_radius ,fc =tmpcolor,ec=tmpcolor,linewidth=0,alpha=alpha)
				self.ax.add_patch(r)
			elif circle_start1 <= startband <= endband <=  circle_end1:
				self.__plothalf(semic_radius,startband,endband,center_x1,center_y1,coff=coff,color=tmpcolor,alpha=alpha)	
			elif cleft <= startband <= endband <= centromere:
				self.__plothalf(semic_radius,startband,endband,cleft,center_y1,coff=coff,color=tmpcolor,alpha=alpha)
			elif centromere <= startband <= endband <= cright:
				self.__plothalf(semic_radius,startband,endband,cright,center_y1,coff=coff,color=tmpcolor,alpha=alpha)
			elif circle_start2 <= startband <= endband <= circle_end2:
				self.__plothalf(semic_radius,startband,endband,center_x2,center_y1,coff=coff,color=tmpcolor,alpha=alpha)
		return 0

	def plot_halfcolorbin(self,datalikekaryo,colordefine,chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,coff=1,alpha=1.0):
		for chrom,start,end,pq,color in datalikekaryo[chrom]:
			tmpcolor = colordefine[color]
			startband = (start+1) / self.unit
			endband   = (end+1)   / self.unit
			center_x1 = circle_end1; center_x2 = circle_start2; center_y1 = ystart + semic_radius
			if circle_end1 <= startband <= circle_start2 and  circle_end1<=endband<=circle_start2:
				r = Rectangle((startband,ystart+0.5*(1+coff)*semic_radius), endband - startband , semic_radius ,fc =tmpcolor,ec=tmpcolor,linewidth=0,alpha=alpha)
				self.ax.add_patch(r)
			elif endband <=  circle_end1:
				self.__plothalf(semic_radius,startband,endband,center_x1,center_y1,coff=coff,color=tmpcolor,alpha=alpha)
			elif startband >= circle_start2:
				self.__plothalf(semic_radius,startband,endband,center_x2,center_y1,coff=coff,color=tmpcolor,alpha=alpha)
			elif startband < circle_end1 and endband > circle_end1:
				self.__plothalf(semic_radius,startband,circle_end1,center_x1,center_y1,coff=coff,color=tmpcolor,alpha=alpha)
				r = Rectangle((circle_end1,ystart+0.5*(1+coff)*semic_radius), endband - circle_end1 , semic_radius ,fc =tmpcolor,ec=tmpcolor,linewidth=0)
				self.ax.add_patch(r)
			elif endband > circle_start2 and startband < circle_start2:
				self.__plothalf(semic_radius,circle_start2,endband,center_x2,center_y1,coff=coff,color=tmpcolor,alpha=alpha)
				r = Rectangle((startband,ystart+0.5*(1+coff)*semic_radius), circle_start2 - startband , semic_radius ,fc =tmpcolor,ec=tmpcolor,linewidth=0)
				self.ax.add_patch(r)
		return 0
	def get_parameter2(self,chrom,ratio=0.03):
		chromosome_length,centromere = self.__get_chrom_len(chrom)
		if centromere == None:
			sys.stderr.write("[ERROR] please use verion1, because lack of centromere information!\n")
			sys.exit(1)
		centromere_band = centromere/self.unit
		chromosome_length_wband = chromosome_length / self.unit
		chromosome_length_wband_1 = self.__get_chrom_len("chr1")[0]/self.unit
		semic_radius = min(chromosome_length_wband_1 * ratio,chromosome_length_wband/10.0)
		circle_start1 =  1/self.unit
		circle_end1   =  circle_start1 + semic_radius
		circle_start2 = chromosome_length_wband - semic_radius
		circle_end2   = chromosome_length_wband
		circle_centromere = centromere_band
		ystart = self.ystart + self.plotorders[chrom] * 2.5 * semic_radius
		return circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,circle_centromere
	def get_parameter(self,chrom,ratio = 0.03):
		chromosome_length,centromere = self.__get_chrom_len(chrom)
		chromosome_length_wband = chromosome_length / self.unit
		chromosome_length_wband_1 = self.__get_chrom_len("chr1")[0]/self.unit
		semic_radius = min(chromosome_length_wband_1 * ratio,chromosome_length_wband/10.0)
		circle_start1 =  1/self.unit
		circle_end1   =  circle_start1 + semic_radius
		circle_start2 = chromosome_length_wband - semic_radius
		circle_end2   = chromosome_length_wband
		ystart = self.ystart + self.plotorders[chrom] * 2.5 * semic_radius
		return circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius
	def plot_frame2(self,chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,centromere):
		chromosome_length = self.__get_chrom_len(chrom)[0]
		chromosome_length_wband = chromosome_length / self.unit
		self.ax.plot([circle_end1,centromere-semic_radius],[ystart,ystart],'-',color='black',linewidth=1.0)
		self.ax.plot([centromere+semic_radius,circle_start2],[ystart,ystart],'-',color='black',linewidth=1.0)
		self.ax.plot([circle_end1,centromere-semic_radius],[ystart + semic_radius * 2,ystart + semic_radius * 2],'-',color='black',linewidth=1.0)
		self.ax.plot([centromere+semic_radius,circle_start2],[ystart + semic_radius * 2,ystart + semic_radius * 2],'-',color='black',linewidth=1.0)
		center_x1 = circle_end1; center_x2 = circle_start2
		center_y1 = ystart + semic_radius
		theta1 = 90; theta3= 270
		theta2 = 270; theta4 = 90
		w1 = Wedge((center_x1, center_y1), semic_radius, theta1, theta2,width=0.0, linewidth=1.0, facecolor='white', edgecolor='black')
		w2 = Wedge((center_x2, center_y1), semic_radius, theta3, theta4,width=0.0, linewidth=1.0, facecolor='white', edgecolor='black')
		w3 = Wedge((centromere+semic_radius,center_y1), semic_radius, theta1, theta2,width=0.0, linewidth=1.0, facecolor='white', edgecolor='black')
		w4 = Wedge((centromere-semic_radius,center_y1), semic_radius, theta3, theta4,width=0.0, linewidth=1.0, facecolor='white', edgecolor='black')
		self.ax.add_patch(w1); self.ax.add_patch(w2); self.ax.add_patch(w3); self.ax.add_patch(w4);
		self.ax.set_xlim(circle_start1-2,chromosome_length_wband+2)
		self.ax.set_ylim(-1,center_y1 + semic_radius + 1)
		return circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,centromere
	def plot_frame(self,chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius): # ratio to plot semicircles
		chromosome_length = self.__get_chrom_len(chrom)[0]
		chromosome_length_wband = chromosome_length / self.unit # whole band of chrom
		self.ax.plot([circle_end1,circle_start2],[ystart,ystart],'-',color='black',linewidth=1.0)
		self.ax.plot([circle_end1,circle_start2],[ystart + semic_radius * 2,ystart + semic_radius * 2],'-',color='black',linewidth=1.0)
		center_x1 = circle_end1; center_x2 = circle_start2
		center_y1 = ystart + semic_radius
		theta1 = 90; theta3= 270
		theta2 = 270; theta4 = 90
		w1 = Wedge((center_x1, center_y1), semic_radius, theta1, theta2,width=0.0, linewidth=1.0, facecolor='white', edgecolor='black')
		w2 = Wedge((center_x2, center_y1), semic_radius, theta3, theta4,width=0.0, linewidth=1.0, facecolor='white', edgecolor='black')
		self.ax.add_patch(w1)	
		self.ax.add_patch(w2)	
		#self.__plothalf(semic_radius,1.0,4.2,center_x1,center_y1,coff=1)
		self.ax.set_xlim(circle_start1-2,chromosome_length_wband+2)
		self.ax.set_ylim(-1,center_y1 + semic_radius + 1)
		return circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius
	def plot_script1(self,chrom="chr3"): # only plot frame
		circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius = self.get_parameter(chrom)
		self.__plotwhole(semic_radius,1.0,4.0,circle_end1,ystart+semic_radius,coff=1,bins=40,color="red",alpha=1.0)
		self.plot_frame(chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius)
		self.save("chr3",fmt="both")
		return 0
	def plot_script4halfcolorbin(self,chrom="chr3",coff=1):
		circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius = self.get_parameter(chrom)
		self.plot_halfcolorbin(self.karyo_dict,Karyocolors,chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius)
		self.plot_frame(chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius)
		self.save("chr3",fmt="both")
		return 0
	def plot_script4karyo(self,chrom="chr3"):
		circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius  = self.get_parameter(chrom)
		self.plot_karyo(chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius)
		self.plot_frame(chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius)
		self.save("chr3",fmt="both")
		return 0
	def plot_script4halfcolorbin2(self,chrom="chr3",coff=1):
		circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,centromere = self.get_parameter2(chrom)
		self.plot_halfcolorbin2(self.karyo_dict,Karyocolors,chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,centromere)
		self.plot_frame2(chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,centromere)
		self.save("chr3",fmt="both")
		return 0
	def plot_script4karyo2(self,chrom="chr3"):
		circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,centromere = self.get_parameter2(chrom)
		self.plot_karyo2(chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,centromere)
		self.plot_frame2(chrom,circle_start1,circle_end1,circle_start2,circle_end2,ystart,semic_radius,centromere)
		self.save("chr3",fmt="both")
		return 0

if __name__ == '__main__':
	karyo_dict = parsekaryo(sys.argv[1])
	plotfig = Plotchrom(karyo_dict)
	#plotfig.plot_script1()
	#plotfig.plot_script4karyo()
	#plotfig.plot_script4halfcolorbin()
	#plotfig.plot_script4karyo2()
	plotfig.plot_script4halfcolorbin2()
	
