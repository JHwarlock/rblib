# -*- coding: UTF-8 -*-
import sys
from rklib.utils import us_sort
from rblib import seqio

"""
Contributors
	- rongzhengqin@basepedia.com
	- zhush@basepedia.com
"""

class Transcript(object):
	"""
	# 0-based [,) 0-based left closed and right open interval
	"""
	def __init__(self,transcriptid,transcriptname="",strand = ".",chrom=None,exons = [], transcript_type=None,leftmost=None,rightmost=None,transcript_source=None,parsed=0,transcript_status=None,proteinid=None):
		self.transcriptid      = transcriptid
		self.proteinid         = proteinid
		self.transcriptname    = transcriptname
		self.chrom             = chrom
		self.exons             = []
		self.strand            = strand
		self.leftmost          = leftmost ## 这里start  codon 和 stop codon 表明的是 [a,b) , 仅仅表示基因组上 coding的region, st a < b; 而非真实的start 和  stop codon的位置
		self.rightmost         = rightmost
		self.transcript_source = transcript_source  ## "gencode"
		#self.otherids         = []
		self.transcript_status = transcript_status ## "known"
		self.utr5              = [] # 针对的是 mRNA 的上下游，而非genomic， 因为有时，我们只是为了获取utr3 或 utr5，但排列的顺序仍然是基于基因组的位置，从小到大，且[a,b) && a < b
		self.utr3              = [] # 均类似上述
		self.cds               = [] # 均类似上述
		self.intron            = [] # 均类似上述
		self.parsed            = 0    # whether cds , utr ,intron was parsed.
		self.exonsorted        = 0    # to record , whether exons was sorted 
		self.merged            = 0    # to record , whether exons was merged  # merge 是避免 exon 和 CDS 的冗余。从而得到非冗余的junction 位置
		self.transcript_type   = transcript_type
		# here we have two choice，one is and， another is or，如果我们选择 and， 那么partial rna就不在我们的考虑范围内
		if leftmost is not None and rightmost is not None: ## if leftmost or rightmost  <> None, cds is kown or partial known. if has, there is self.cds  and  self.intro and self.utr  regions # but note , it include both partial and complete cds region , 那么对于partial cds，这个CDS 是 5‘ partial 还是3’ partial ， 由self.leftmost None? self.rightmost None ?，以及 strand 信息来判定。 对于cdsknown ，还要解析CDS 的 codon frame，0,1,2 。 因此，对于partial的解析 就要变得非常的小心
			self.cdsknown = 1
		else:
			self.cdsknown = 0
		self.cdsstatus = [] # for gtf rules
		self.cdsstatus2 = [] # for refgene.txt(UCSC) rules
	def __str__(self):
		"""just for print to describe transcript info"""
		strout = """---
transcript id   : %s
transcript name : %s
transcript type : %s
chrom           : %s
strand          : %s
exons           : %s
intron          : %s
coding region   : %s ~ %s
cds             : %s
# 0-based left closed and right open interval\n---\n"""%(self.transcriptid,self.transcriptname,self.transcript_type,self.chrom,self.strand,self.exons,self.intron,self.leftmost,self.rightmost,self.cds)
		return strout	
	
	def merge_exons(self):
		"""
		merge for delete redundant exon and cds regions 
		"""
		if self.merged: return 0
		if not self.exonsorted: self.sortexon()
		self.exons = seqio.merge_region(self.exons)
		self.merged = 1
		return 0

	def add_exon(self,start,end): # add to a transcript instance 
		assert start < end #  这里其实有个问题，添加 start 和 end 时，一定要保证 start  < end, 且这里是 0-based [a,b) , 若gtf 和 gff 类的1-based 坐标体系，strand 为 - 时，start  > end ，一定要交换后，再对小的坐标 -1 
		if self.parsed:
			sys.stderr.write("[WARN] %s has been parsed, please addexon before gene parsed\n"%self.transcriptid)
			return 0
		self.exons.append([start,end])
		return 0

	def sortexon(self):# 排序和去冗余（即merge） 对正确parse 非常重要！！！
		if self.exonsorted: return 0
		self.exons = us_sort(self.exons,0,1)
		self.exonsorted = 1
		return 0
	
	def parse_transcript(self): # parse to get utr, intron ,cds  ...
		if self.parsed:
			#sys.stderr.write("[INFO] %s has been parsed\n"%self.transcriptid)
			return 0
		if not self.merged: self.merge_exons() # sort and merge for self.exons
		strand = self.strand
		sortexons = self.exons
		introns = []; UTR_f = []; UTR_b = []; CDS = []
		exon_num = len(sortexons)
		start = sortexons[0][0] 
		cdslen = 0
		if self.leftmost == None:cds_start = start # 这组 if else 绝对不能修改self 下的leftmost 和 rightmost
		else: cds_start = self.leftmost
		if self.rightmost == None: cds_stop   = sortexons[-1][-1]
		else: cds_stop  = self.rightmost

		for i in range(exon_num):
			starttmp  = sortexons[i][0]
			endtmp    = sortexons[i][1]
			try:
				introns.append([sortexons[i][1],sortexons[i+1][0]])
			except (ValueError,IndexError): pass
			if not self.cdsknown: continue ## for no cds information, utr and cds ignored 
			if endtmp <= cds_start:
				UTR_f.append([starttmp,endtmp])
				continue
			else:
				if starttmp < cds_start:
					UTR_f.append([starttmp,cds_start])
					starttmp = cds_start
			if starttmp <= cds_stop < endtmp:
				UTR_b.append([cds_stop,endtmp])
				endtmp = cds_stop
			if starttmp > cds_stop:
				UTR_b.append([starttmp,endtmp])
				continue
			CDS.append([starttmp,endtmp])
			cdslen += endtmp - starttmp
		self.intron  = introns 
		self.parsed = 1
		if not self.cdsknown: return 0
		self.cds     = CDS
		### 需要判断cdsstatus
		cdsstatus = [".",] * len(CDS)
		cdsstatus2 = ["-1",] * len(CDS)
		####
		if strand == "+":
			if self.leftmost is not None:
				cdsstatus[0] = 0; cdsstatus2[0] = 0
			else: # 当leftmost 缺失，可能是个3‘partial CDS， 那么需要确定第一个CDS frame 即可确定后续的frame
				cdsstatus[0] = cdslen % 3; cdsstatus2[0] = [0,2,1][cdsstatus[0]]
			for i in range(len(CDS)-1):
				cdsstatus[i+1] = (cdsstatus[i] - (CDS[i][-1]-CDS[i][0]))%3 
				cdsstatus2[i+1] = (cdsstatus2[i] + (CDS[i][-1]-CDS[i][0]))%3

			self.utr5 = UTR_f  ## utr all frame are '-1' 
			self.utr3 = UTR_b  

		elif strand == "-":
			if self.rightmost is not None:
				cdsstatus[-1] = 0; cdsstatus2[-1] = 0
			else:
				cdsstatus[-1] = cdslen % 3; cdsstatus2[-1] = [0,2,1][cdsstatus[-1]]
			for i in range(len(CDS)-1,0,-1):
				cdsstatus[i-1] = (cdsstatus[i] - (CDS[i][-1]-CDS[i][0]))%3
				cdsstatus2[i-1] = (cdsstatus2[i] + (CDS[i][-1]-CDS[i][0]))%3
			self.utr5 = UTR_b
			self.utr3 = UTR_f
		
		else:
			self.utr5 = UTR_f
			self.utr3 = UTR_b
			sys.stderr.write("[INFO] %s :Unkown strand!\n"%self.transcriptid)
		self.cdsstatus = cdsstatus
		self.cdsstatus2 = cdsstatus2
		return 0

class Gene(object):
	def __init__(self,geneid,genename="",gene_type=None,gene_source=None,gene_status=None):
		self.geneid          = geneid
		self.genename        = genename
		self.gene_type       = gene_type
		self.gene_source     = gene_source
		self.gene_status     = gene_status
		self.transcripts     = {} ## transcriptid as key
		self.gene_start      = None
		self.gene_stop       = None
		self.chrom           = None
		self.strand          = None
		self.mergedexons      = []
		self.parsed          = 0 # use to control its transcripts parsed 
	def __str__(self):
		strout = """================
gene id     : %s
gene name   : %s
gene type   : %s
include transcripts : %s\n================\n"""%(self.geneid,self.genename,self.gene_type,self.transcripts.keys())
		for i in self.transcripts:
			strout += str(self.transcripts[i])
		return strout	
	
	def get_transcript(self,tid,transcriptname="",strand = ".",chrom=None,exons = [], transcript_type=None,leftmost=None,rightmost=None,transcript_source=None,parsed=0,transcript_status=None,proteinid=None):
		if tid not in self.transcripts:
			tmptranscript = Transcript(tid,transcriptname=transcriptname,strand=strand,chrom=chrom,exons=exons,transcript_type=transcript_type,leftmost=leftmost,rightmost=rightmost,transcript_source=transcript_source,parsed=parsed,transcript_status=transcript_status,proteinid=proteinid)
			self.add_transcript(tmptranscript)
		return self.transcripts[tid]

	def add_transcript(self,transcript):
		assert isinstance(transcript,Transcript)
		if transcript.transcriptid not in self.transcripts:
			self.transcripts[transcript.transcriptid] = transcript
		else:
			sys.stderr.write("[WARN] same transcript id: '%s'\n"%transcript.transcriptid)
		return self.transcripts[transcript.transcriptid]

	def parse_gene(self):
		if self.parsed: return 0
		if not self.transcripts:
			sys.stderr.write("[WARN] check '%s', no transcript found\n"%self.geneid)
			return 0
		self.chrom = 0 # choose first transcrpt as, here we should add it 
		allexons = []
		for tid in self.transcripts:
			tmptranscript = self.transcripts[tid]
			tmptranscript.parse_transcript()
			self.chrom = tmptranscript.chrom
			self.strand = tmptranscript.strand
			allexons += tmptranscript.exons
			#if self.gene_start == None or self.gene_start > tmptranscript.exons[0][0]:self.gene_start = tmptranscript.exons[0][0]
			#if self.gene_stop  == None or self.gene_stop  < tmptranscript.exons[-1][-1]: self.gene_stop = tmptranscript.exons[-1][-1]
		allexons =  us_sort(allexons,0,1)
		#print(allexons)
		self.mergedexons = seqio.merge_region(allexons)
		#print(self.mergedexons)
		self.gene_start = self.mergedexons[0][0]
		self.gene_stop  = self.mergedexons[-1][-1]
		self.parsed = 1
		return 0
	def __gtfattr(self,obj,name):
		if obj is not None:
			return ' %s "%s";'%(name,obj)
		else:return ""
	def togtf(self):
		"this method written by zhush, contact: zhush@basepedia.com, modified by rongzhengqin@basepedia.com"
		strout=''
		if not self.parsed: self.parse_gene()
		if not self.parsed: return strout
		Line="""%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\n"""
		geneid            = self.geneid
		geneattr = 'gene_id "%s";'%geneid # gene_name "%s"; gene_type "%s";'
		geneattr += self.__gtfattr(self.genename,"gene_name")
		geneattr += self.__gtfattr(self.gene_type,"gene_type")
		gene_start        = self.gene_start+1
		gene_stop         = self.gene_stop
		gene_source       = "." if self.gene_source == None else self.gene_source
		chrom             = self.chrom
		strand            = self.strand
		strout += Line%(chrom,gene_source,"gene",gene_start,gene_stop,".",strand,".",geneattr)

		for transcript in self.transcripts:
			assert isinstance(self.transcripts[transcript],Transcript)
			transcriptInstance = self.transcripts[transcript]
			
			tid               = transcriptInstance.transcriptid
			attr =  geneattr
			attr += self.__gtfattr(tid,"transcript_id")
			attr += self.__gtfattr(transcriptInstance.transcriptname,"transcript_name")
			attr += self.__gtfattr(transcriptInstance.transcript_status,"transcript_status")
			attr += self.__gtfattr(transcriptInstance.transcript_type,"transcript_type")
			attr += self.__gtfattr(transcriptInstance.proteinid,"protein_id")
			chrom             = transcriptInstance.chrom
			exons             = transcriptInstance.exons
			strand            = transcriptInstance.strand
			transcript_source = "." if transcriptInstance.transcript_source == None else transcriptInstance.transcript_source
			cds               = transcriptInstance.cds
			intron            = transcriptInstance.intron
			cdsstatus         = transcriptInstance.cdsstatus
			## first to export transcript region
			strout += Line%(chrom,transcript_source,"transcript",exons[0][0]+1,exons[-1][-1],".",strand,".",attr)
			
			if transcriptInstance.cdsknown and strand in ["+","-"]:
				if transcriptInstance.leftmost is not None:
					s = transcriptInstance.leftmost + 1
					e = s + 2
					annoregion = "start_codon" if strand == "+" else "stop_codon"
					strout += Line%(chrom,transcript_source,annoregion,s,e,".",strand,"0",attr)
				if transcriptInstance.rightmost is not None:
					e = transcriptInstance.rightmost
					s = e - 2
					annoregion = "start_codon" if strand == "-" else "stop_codon"
					strout += Line%(chrom,transcript_source,annoregion,s,e,".",strand,"0",attr)
				for i in range(len(cds)): # however partial or whole， CDS should  export
					s = cds[i][0] + 1
					e = cds[i][-1]
					tmpstatus = cdsstatus[i]
					strout += Line%(chrom,transcript_source,"CDS",s,e,".",strand,tmpstatus,attr)
			for s,e in exons:
				s+=1
				strout += Line%(chrom,transcript_source,"exon",s,e,".",strand,".",attr)
		return strout

	def torefgene(self):
		"this method written by zhush, contact: zhush@basepedia.com, modified by rongzhengqin@basepedia.com"
		"Note : in refgene format from UCSC, only output transcripts"
		strout=''
		if not self.parsed: self.parse_gene()
		if not self.parsed: return strout
		genename           = "" if self.genename == None else self.genename
		for transcript in self.transcripts:
			assert isinstance(self.transcripts[transcript],Transcript)
			transcriptInstance = self.transcripts[transcript]
			tid               = transcriptInstance.transcriptid
			#tname             = "" if  transcriptInstance.transcriptname
			chrom             = transcriptInstance.chrom
			exons             = transcriptInstance.exons
			strand            = transcriptInstance.strand
			transcript_source = transcriptInstance.transcript_source
			transcript_status = transcriptInstance.transcript_status
			outexonstarts = []
			outexonstops  = []
			outexonstatus = []
			num = 0
			if transcriptInstance.cdsknown:
				utr1 = transcriptInstance.utr5 if strand == "+" else transcriptInstance.utr3
				utr2 = transcriptInstance.utr3 if strand == "+" else transcriptInstance.utr5
				cds  = transcriptInstance.cds
				cdsstatus = transcriptInstance.cdsstatus2 # use refgene CDS status rules
				for s,e in utr1:
					outexonstarts.append(str(s)); outexonstops.append(str(e)); outexonstatus.append("-1"); num += 1
				for i in range(len(cds)):
					s = cds[i][0]
					e = cds[i][1]
					tmpstatus = cdsstatus[i]
					outexonstarts.append(str(s)); outexonstops.append(str(e)); outexonstatus.append(str(tmpstatus)); num += 1
				for s,e in utr2:
					outexonstarts.append(str(s)); outexonstops.append(str(e)); outexonstatus.append("-1"); num += 1
			else:
				for s,e in exons:
					outexonstarts.append(str(s)); outexonstops.append(str(e)); outexonstatus.append("-1"); num += 1
			if transcriptInstance.leftmost is not None or transcriptInstance.rightmost is not None:
				info1 = "cmpl"; info2 = "cmpl"
				leftmost = transcriptInstance.leftmost; rightmost = transcriptInstance.rightmost;
				if transcriptInstance.leftmost == None:
					info1 = "incmpl"
					leftmost = exons[0][0]
				if transcriptInstance.rightmost  == None:
					info2 = "incmpl"
					rightmost  = exons[-1][-1]
			else:
				info1 = "unk"; info2 = "unk"
				leftmost = exons[-1][-1]; rightmost  = exons[-1][-1] # 即没有 coding region
				
			Line = "\t".join(["1",tid,chrom,strand,str(exons[0][0]),str(exons[-1][-1]),str(leftmost),str(rightmost),str(num),",".join(outexonstarts)+",",",".join(outexonstops)+",","0",genename,info1,info2,",".join(outexonstatus)+","])
			strout += Line + "\n"
		return strout


from rblib.seqprocess import Key_obj

## use to parse UCSC refgene.txt
def readrefgene(refgenefile): ## read refgene format
	hgene = {}
	for ret in seqio.refgene_parse(refgenefile):
		num,nm_name,chrom,strand,exon_s,exon_e,cds_s,cds_e,exon_num,exonstarts,exonends,uniq_id,symbol, kown1, kown2, exon_status =  ret
		
		tmpgene = hgene.setdefault(symbol,Gene(symbol,symbol))
		cds_s = int(cds_s) if kown1 == "cmpl" else None
		cds_e = int(cds_e) if kown2 == "cmpl" else None
		tmptranscript = tmpgene.get_transcript(nm_name,transcriptname="",strand = strand,chrom=chrom,exons = [],leftmost=cds_s,rightmost=cds_e)
		exonstarts = exonstarts.split(",")
		exonends   = exonends.split(",")
		for i in range(int(exon_num)):
			tmptranscript.add_exon(int(exonstarts[i]),int(exonends[i]))
		#tmptranscript.sortexon()
		tmptranscript.parse_transcript()
		#tmpgene.add_transcript(tmptranscript)
	
	for key in hgene:
		hgene[key].parse_gene() ## parsed ok
	return hgene

# use for gtf only now , use to read GTF files
class Annoregion2(object): 
	def __init__(self,fmt="gtf",gattr="gene_name",tattr="transcript_name",gidattr="gene_id",tidattr="transcript_id",genetypeattr="gene_type",transcripttypeattr="transcript_type",tanno="exon,CDS,UTR",codonanno_start = "start_codon",codonanno_stop="stop_codon",pidattr="protein_id"):
		self.fmt = fmt
		self.gattr = Key_obj(gattr)
		self.tattr = Key_obj(tattr)
		self.codonanno_start = dict.fromkeys(codonanno_start.split(","),None)
		self.codonanno_stop  = dict.fromkeys(codonanno_stop.split(","),None)
		self.gidattr = Key_obj(gidattr)
		self.tidattr = Key_obj(tidattr)
		self.pidattr = Key_obj(pidattr)
		self.genetypeattr = Key_obj(genetypeattr)
		self.transcripttypeattr = Key_obj(transcripttypeattr)
		self.tanno  = dict.fromkeys(tanno.split(","),None)  ## target annotation region
		self.h = {} #?  h :  geneid => gene_instance
		self.biotypes = {} # record how many biotypes  ?? need or not ?

	def __get_codon(self,start,end,gid,gname,tid,tname,chrom,strand,genetype,transcripttype,based=1,startanno = 1,proteinid=None):
		# 返回的永远是最大的coding region， 即 [small, large) 的coding region, 与 refgene 匹配
		# if end < start: start,end = end,start
		try:
			assert strand in ["+","-"]
		except:
			return 0
		start = start-based
		tmpgene = self.h.setdefault(gid,Gene(gid,gname,gene_type=genetype))
		
		tmptranscript = tmpgene.get_transcript(tid,transcriptname=tname,strand = strand,chrom=chrom,transcript_type=transcripttype,proteinid=proteinid)
		#tmptranscript = self.h[gid].transcripts.setdefault(tid,Transcript(tid,transcriptname=tname,strand = strand,chrom=chrom,transcript_type=transcripttype))
		if (startanno and strand == "-")  or ((not startanno) and strand == "+"):
			tmptranscript.rightmost = end if tmptranscript.rightmost  == None else max(tmptranscript.rightmost,end)
		else:
			tmptranscript.leftmost = start if tmptranscript.leftmost == None else min(tmptranscript.leftmost,start)
		if tmptranscript.rightmost is not None and tmptranscript.leftmost is not None: tmptranscript.cdsknown = 1
		return 0

	def gtf2exons(self,gtffn):
		f = open(gtffn,"r")
		for line in f:
			if line.startswith("#"):continue
			chrom,source,region_type,start,end,score,strand,codon,commnet = line.rstrip("\n").split("\t")
			gname = self.gattr.getvalue(commnet)
			tname = self.tattr.getvalue(commnet)
			gid   = self.gidattr.getvalue(commnet)
			tid   = self.tidattr.getvalue(commnet)
			pid   = self.pidattr.getvalue(commnet)
			genetype = self.genetypeattr.getvalue(commnet)
			transcripttype = self.transcripttypeattr.getvalue(commnet)
			start = int(start)
			end   = int(end)
			if end < start: start,end = end,start # must be sort first
			if region_type in self.tanno:
				tmpgene = self.h.setdefault(gid,Gene(gid,gname,gene_type=genetype))
				tmptranscript = self.h[gid].transcripts.setdefault(tid,Transcript(tid,transcriptname=tname,strand = strand,chrom=chrom,transcript_type=transcripttype,proteinid=pid))
				tmptranscript.add_exon(start-1,end)

			if region_type in self.codonanno_start:
				self.__get_codon(start,end,gid,gname,tid,tname,chrom,strand,genetype,transcripttype,proteinid=pid)
			if region_type in self.codonanno_stop:
				self.__get_codon(start,end,gid,gname,tid,tname,chrom,strand,genetype,transcripttype,startanno=0,proteinid=pid)

		f.close()
		for gid in self.h:
			self.h[gid].parse_gene()  ## parsed ok
		return 0

	def __str__(self):
		strout = ""
		for gid in self.h:
			strout += str(self.h[gid])
			break
		strout += "... ..."
		return strout

def gene2file(hgene,fmt="gtf",outputprefix = "test."):
	# brefore parse
	assert fmt in ["gtf","refgene"]
	outputfn = outputprefix + fmt
	fout = open(outputfn,"w")
	if fmt == "gtf":
		for geneid in hgene:
			fout.write(hgene[geneid].togtf())
	elif fmt == "refgene":
		for geneid in hgene:
			fout.write(hgene[geneid].torefgene())
	fout.close()
	sys.stderr.write("[INFO] file output to '%s'\n"%outputfn)
	return 0

if __name__ == "__main__":
	#readrefgene(sys.argv[1])

	## read GenCode gtf 
	if sys.argv[1].endswith(".gtf"):
		annoregion2 = Annoregion2(fmt="gtf")
		annoregion2.gtf2exons(sys.argv[1])
		gene2file(annoregion2.h,fmt="refgene")
		gene2file(annoregion2.h,fmt="gtf")
	## read refgene.txt
	if sys.argv[1].endswith(".refgene"):
		hgene = readrefgene(sys.argv[1])
		gene2file(hgene,fmt="refgene")
		gene2file(hgene,fmt="gtf")
