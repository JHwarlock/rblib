# -*- coding: UTF-8 -*-
import sys
from rblib import seqio
from rklib.utils import us_sort
from rblib.defs import hgeneclass
from rblib import bamio
# 0-based [,)   return regions  1-based coor like  gff
def refgene_parse_transcript(chrom,start,end,cds_start,cds_stop,exon_num,strand,exon_starts,exon_ends,exon_status):
	exons = []
	introns = []
	UTR_f = []
	UTR_b = []
	CDS = []
	exon_starts_arr = exon_starts.split(",")
	exon_ends_arr = exon_ends.split(",")
	start +=1; cds_start += 1;
	regions = None
	for i in range(exon_num):
		exons.append(chrom+"_"+str(int(exon_starts_arr[i])+1)+"_"+exon_ends_arr[i])
		try:
			introns.append(chrom+"_"+str(int(exon_ends_arr[i])+1)+"_"+str(int(exon_starts_arr[i+1])))
		except (ValueError,IndexError): pass
		if exon_status[i] == "-1":
			if int(exon_ends_arr[i]) <= cds_start:
				UTR_f.append(chrom+"_"+str(int(exon_starts_arr[i])+1)+"_"+exon_ends_arr[i])
			else:
				UTR_b.append(chrom+"_"+str(int(exon_starts_arr[i])+1)+"_"+exon_ends_arr[i])
		else:
			starttmp = int(exon_starts_arr[i])+1
			endtmp = int(exon_ends_arr[i])
			if endtmp < cds_start:
				UTR_f.append(chrom+"_"+str(int(exon_starts_arr[i])+1)+"_"+exon_ends_arr[i])
				continue
			else:
				if starttmp < cds_start:
					UTR_f.append(chrom+"_"+str(int(exon_starts_arr[i])+1)+"_"+str(cds_start-1))
					starttmp = cds_start
			if cds_stop < endtmp and starttmp <= cds_stop:
				UTR_b.append(chrom+"_"+str(int(cds_stop)+1)+"_"+exon_ends_arr[i])
				endtmp = cds_stop
			if starttmp > cds_stop:
				UTR_b.append(chrom+"_"+str(int(exon_starts_arr[i])+1)+"_"+exon_ends_arr[i])
				continue
			CDS.append(chrom+"_"+str(starttmp)+"_"+str(endtmp))
	if strand == "+":
		regions = [exons,introns,UTR_f,UTR_b,CDS]
	elif strand == "-":
		regions = [exons,introns,UTR_b,UTR_f,CDS]
	else:
		sys.stderr.write("Unkown strand!\n")
	return regions

class Comment_obj(object):
	def __init__(self,comment):
		self.comment = comment
	def getvalue(self,key):
		ret = self.comment.split(key,1)
		defaut_return = "-"
		if len(ret) >= 2:
			defaut_return = ret[-1].strip().split(";")[0].strip().strip("\"")
		return defaut_return
	## com_ins = Comment_obj(comment_s)
	## transcript_id_s = com_ins.getvalue("transcript_id ")
class Key_obj(object):
	def __init__(self,key):
		self.key = key
	def getvalue(self,comment):
		ret = comment.split(self.key,1)
		defaut_return = "-"
		if len(ret) >= 2:
			defaut_return = ret[-1].strip().split(";")[0].strip().strip("\"")
		return defaut_return

def fmtMycustom(FileName, Type):
	TmpName = FileName + '.fmt'; ChromNames = []
	with open(FileName, 'r') as fp, open(TmpName, 'w') as fp1:
		for index, line in enumerate(fp):
			if line.startswith('#'): continue
			chrom, start, end, strand, \
			gene_id, trans_id, gene_name, \
			overLapRefName, overLapGenetype, \
			ref_genetype, predict_genetype = line.strip().split('\t')
			if ref_genetype == "-":
				bio_type = predict_genetype
			elif predict_genetype == "-":
				bio_type = hgeneclass[ref_genetype]
			else:
				bio_type = "unknown"
			if chrom not in ChromNames: ChromNames.append(chrom)
			Commnets ="""gene_id "{}";
			transcript_id "{}";
			exon_number "{}";
			gene_name "{}";
			overlap_ref_name "{}";
			overlap_gene_type "{}";
			bio_type "{}";
			p_id "{}";""".format(gene_id, trans_id, index + 1, gene_name, \
					overLapRefName, overLapGenetype, bio_type, 'P_X')
			Commnets = ' '.join(Commnets.split()) 
			String = '\t'.join(map(str, [chrom, Type, 'exon', start, end, '.', strand, '.', Commnets]))
			fp1.write(String + '\n')
	sortName = TmpName + '.sort'
	with open(TmpName, 'r') as fp, open(sortName, 'w') as fp1:
		Words = us_sort([[line.split('\t', 5)[0], \
				line.split('\t', 5)[1], \
				line.split('\t', 5)[2], \
				int(line.split('\t', 5)[3]), \
				int(line.split('\t', 5)[4]), \
				line.split('\t', 5)[5]] \
				for line in fp], 0, 3, 4)
		[fp1.write('\t'.join(map(str, line))) for line in Words]
	bamio.gtfgffdb(sortName)
	return sortName

"""
plot_symbols_exons = [["AT01","ATname1","chr1",10040,14750,"-",[[10020,14820],],"coding"],["AT02","ATname2","chr1",10045,14802,"-",[[10020,12100],[13141,14890]],       "lnc"]]
基因名往往在基因组上不唯一，因此，gtf and gff: hash 应采用geneid 作为key; refgene: geneid + chrom + start + end
"""
class Annoregion(object):
	def __init__(self,fmt="gtf",gattr="",tattr="",csanno="",ceanno="",gidattr="",tidattr="",biotypeattr="",tanno="exon"):
		self.fmt = fmt
		self.gattr = Key_obj(gattr)
		self.tattr = Key_obj(tattr)
		self.csanno = csanno
		self.ceanno = ceanno
		self.gidattr = Key_obj(gidattr)
		self.tidattr = Key_obj(tidattr)
		#print tidattr
		self.biotypeattr = Key_obj(biotypeattr)
		self.tanno  = tanno
		self.h = {}
		self.biotypes = {}
	def __get_codon(self,start,end,gid,gname,tid,tname,chrom,strand,biotype,based=1,pos=3):
		## 返回的都是小的坐标，因此无论是 + 还是 - ，返回都是 ret ， ret+ 3 # 且返回的是 0 based 坐标。
		start = int(start);end = int(end)
		if end < start: start,end = end,start
		start = start-based
		gidname = gid + "|" + gname
		#print gidname,tid
		if gidname not in self.h:
			self.h[gidname] = [gname,{}]
		if tid not in self.h[gidname][1]:
			self.h[gidname][1][tid] = [tid,tname,chrom,0,0,strand,[[start,end]],biotype]
		else:
			self.h[gidname][1][tid][6].append([start,end])
		postmp = self.h[gidname][1][tid][pos]
		if	postmp == 0:
			self.h[gidname][1][tid][pos] = start
		else:
			if pos == 3:
				self.h[gidname][1][tid][pos] = min(self.h[gidname][1][tid][pos],start)
			elif pos == 4:
				self.h[gidname][1][tid][pos] = max(self.h[gidname][1][tid][pos],start)
	def __getret(self):
		ret = {}
		for gidname in self.h:
			#print gid
			tids = self.h[gidname][1].values()
			ret[gidname] = [self.h[gidname][0],tids[:]]

		return ret
	def gtf2exons(self,gtflines):
		for line in gtflines:
			chrom,source,region_type,start,end,score,strand,codon,commnet = line.rstrip("\n").split("\t")
			#print strand
			gname = self.gattr.getvalue(commnet)
			tname = self.tattr.getvalue(commnet)
			gid   = self.gidattr.getvalue(commnet)
			tid   = self.tidattr.getvalue(commnet)
			biotype = self.biotypeattr.getvalue(commnet)
			gidname = gid + "|" + gname
			#print self.h
			#print gname,tname,gid,tid,biotype,region_type
			if region_type == self.tanno:
				#print region_type,gidname,tid
				
				if gidname not in self.h:
					self.h[gidname] = [gname,{}]
				if tid not in self.h[gidname][1]:
					start = int(start)
					end   = int(end)
					if end < start: 
						start,end = end,start
					start -= 1
					self.h[gidname][1][tid] = [tid,tname,chrom,0,0,strand,[[int(start),int(end)],],biotype] # 返回左闭右开
					self.biotypes[biotype] = None
				else:
					self.h[gidname][1][tid][6].append([int(start)-1,int(end)])
			elif region_type == self.csanno:
				self.__get_codon(start,end,gidname,gname,tid,tname,chrom,strand,biotype)
			elif region_type == self.ceanno:
				self.__get_codon(start,end,gidname,gname,tid,tname,chrom,strand,biotype,pos=4)
		#self.getret()
		#for gid in self.h:
		#	tids= self.h[gid][1].keys()
		#	for i in tids:
		#		print self.h[gid][1][i][0]
		ret =  self.__getret(),self.biotypes.keys()[:]
		return ret
		
					
"""
GTF and GFF:
Fields

Fields must be tab-separated. Also, all but the final field in each feature line must contain a value; "empty" columns should be denoted with a '.'

seqname - name of the chromosome or scaffold; chromosome names can be given with or without the 'chr' prefix. Important note: the seqname must be one used within Ensembl, i.e. a standard chromosome name or an Ensembl identifier such as a scaffold ID, without any additional content such as species or assembly. See the example GFF output below.
source - name of the program that generated this feature, or the data source (database or project name)
feature - feature type name, e.g. Gene, Variation, Similarity
start - Start position of the feature, with sequence numbering starting at 1.
end - End position of the feature, with sequence numbering starting at 1.
score - A floating point value.
strand - defined as + (forward) or - (reverse).
frame - One of '0', '1' or '2'. '0' indicates that the first base of the feature is the first base of a codon, '1' that the second base is the first base of a codon, and so on..
attribute - A semicolon-separated list of tag-value pairs, providing additional information about each feature.
Sample GTF output from Ensembl data dump:
1 transcribed_unprocessed_pseudogene  gene        11869 14409 . + . gene_id "ENSG00000223972"; gene_name "DDX11L1"; gene_source "havana"; gene_biotype "transcribed_unprocessed_pseudogene"; 
1 processed_transcript                transcript  11869 14409 . + . gene_id "ENSG00000223972"; transcript_id "ENST00000456328"; gene_name "DDX11L1"; gene_sourc e "havana"; gene_biotype "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-002"; transcript_source "havana";

"""
