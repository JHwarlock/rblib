# -*- coding: UTF-8 -*-
import sys
import os
import shelve as s
import numpy as np
#建立简单的数字索引，针对单个位置和range 进行注释搜索，获取数据等等

class POSIDX(object):
	def __init__(self,filein):
		self.filein = filein
		self.fidx   = self.filein+".idx"
		self.fpin   = file(self.filein,"r")
		self.fpidx  = None
		self.hidx   = {}
		self.binsize = 0
	def checkidx(self):
		#try:
		#	self.hidx = s.open(self.fidx)
		#	name,binsize,fsize,ct,mt = self.hidx["header"]
		#except:
		#	return 1
		try:
			self.fpidx = file(self.fidx,"r")
		except:return 1
		self.fpidx.seek(0)
		header = self.fpidx.next().rstrip("\n")
		self.fpidx.seek(0)
		if not header.startswith("#"):return 1
		try:
			name,binsize,fsize,ct,mt = header.split("\t")
			#print "###########",name,binsize,fsize,ct,mt
		except:
			return 1
		info = os.stat(self.filein)
		fsizetmp = info.st_size
		f_mt = info.st_mtime
		f_ct = info.st_ctime
		if int(fsize) == int(fsizetmp) and int(ct) == int(f_ct) and int(mt) == int(f_mt):
			sys.stderr.write("[INFO] FILE '%s' has indexed\n"%self.filein)
			return 0
		else:
			return 1
	def buildidx(self,posidx=1,posidx1=1,binsize=10000):
		if self.checkidx():pass
		else:
			sys.stderr.write("[INFO] IDX exist.\n")
			return 0
		try:
			os.remove(self.fidx)
		except:pass
		self.hidx = {}
		name = "#POSIDX"
		info = os.stat(self.filein)
		fsizetmp = info.st_size
		f_mt = info.st_mtime
		f_ct = info.st_ctime
		#self.hidx["header"] = [name,binsize,fsizetmp,f_ct,f_mt]
		self.fpidx = file(self.fidx,"w")
		self.fpidx.write("\t".join([name,str(binsize),str(int(fsizetmp)),str(int(f_ct)),str(int(f_mt))])+"\n")
		self.fpin.seek(0)
		lines_c = 0
		while 1:
			fcurrent = self.fpin.tell()
			line = self.fpin.readline()
			lines_c += 1
			if lines_c%(500000) == 0:sys.stderr.write("[INFO] Process %d lines\n"%lines_c)
			if not line:break
			if line.startswith("#"):continue
			arr = line.rstrip().split("\t")
			if len(arr) < posidx1+1:continue
			try:
				chrom = arr[0]
				pos = arr[posidx]
				pos1 = arr[posidx1]
				bintmp = int(pos)/binsize
				bintmp1 = int(pos1)/binsize
			except:
				print arr
				continue
			if chrom in self.hidx:
				if bintmp not in self.hidx[chrom]:
					self.hidx[chrom][bintmp] = [fcurrent,fcurrent]
			#		print fcurrent
				else:
					self.hidx[chrom][bintmp][1] = fcurrent
			#		print fcurrent
				if bintmp <> bintmp1:
					if bintmp1 not in self.hidx[chrom]:
						self.hidx[chrom][bintmp1] = [fcurrent,fcurrent]
					else:
						self.hidx[chrom][bintmp1][1] = fcurrent
			else:
				sys.stderr.write("[INFO] Build for %s\n"%chrom)
				self.hidx[chrom] = {}
				self.hidx[chrom][bintmp] = [fcurrent,fcurrent]
				if bintmp1 <> bintmp:
					self.hidx[chrom][bintmp1] = [fcurrent,fcurrent]
		self.fpin.seek(0)
		for chrom in self.hidx:
			for bintmp in self.hidx[chrom]:
				out = [chrom,bintmp]+self.hidx[chrom][bintmp]
				self.fpidx.write("\t".join(map(str,out))+"\n")
		self.fpidx.close()
		self.hidx["header"] = [name,binsize,fsizetmp,f_ct,f_mt]
		print self.hidx["header"]
		#self.hidx.close()
		sys.stderr.write("[INFO] IDX Build Done.\n")
		return 0
	def readidx(self):
		if self.checkidx() == 0:
			sys.stderr.write("[INFO] IDX read\n")
		else:
			sys.stderr.write("[ERROR] IDX not found or error.\n")
			sys.exit(1)
		self.fpin.seek(0)
		#self.hidx = s.open(self.fidx)
		#print self.hidx["header"]
		#self.binsize = self.hidx["header"][1]
		self.fpidx = file(self.fidx,"r")
		for line in self.fpidx:
			if line.startswith("#"):
				name,binsize,fsize,ct,mt = line.rstrip("\n").split("\t")
				self.hidx["header"] = [name,int(binsize),int(fsize),int(ct),int(mt)]
				self.binsize = self.hidx["header"][1]
				continue
			chrom,bintmp,start,end = line.split("\t")
			if chrom not in self.hidx:
				self.hidx[chrom] = {}
			self.hidx[chrom][int(bintmp)] = [long(start),long(end)]
		self.fpidx.close()
		return self.hidx
	def search_region(self,chrom,sstart,send):
		#print chrom,sstart,send
		if chrom not in self.hidx: return None
		sstart_t = sstart/self.binsize
		send_t = send / self.binsize
		filestart = np.inf
		fileend = 0 
		for bintmp in xrange(sstart_t,send_t+1):
			#print "bintmp",bintmp
			if bintmp in self.hidx[chrom]:
			#	print "into file idx"
				tmpstart,tmpend = self.hidx[chrom][bintmp]
			#	print "loop",tmpstart,tmpend
				if filestart > tmpstart:filestart = tmpstart
				if fileend < tmpend:fileend = tmpend
		if filestart == np.inf and fileend == 0:
			return None
		else:
			return filestart,fileend
	def close(self):
		del self.hidx
		self.fpin.close()
### first check, then build, read for doing

