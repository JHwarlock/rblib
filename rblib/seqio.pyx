import sys
from rklib import compress
import os
from Bio import SeqIO
from Bio.SeqUtils import GC
from Bio.Seq import Seq
from rklib import utils
import pysam

def fmtseq(string):
	return Seq(string)

def rcseq(string):
	a = Seq(string)
	return str(a.reverse_complement())

def faidx(genome):
	# Random access to fasta formatted files that have been indexed by faidx.
	# Note that region strings are 1-based, while start and end denote an interval in python coordinates. The
	# region is specified by reference, start and end.
	tmpidx = pysam.FastaFile(genome)
	return tmpidx

def seqfromgenome(faidx,chrom,start,end):
	seq_retrieve = faidx.fetch(chrom,start,end)
	return seq_retrieve

def cal_GC(seq):
	return GC(seq)

def variant_snpindel_pop(total_fn):
	f = compress.gz_file(total_fn,"r")
	for line in f:
		if line.startswith("#"):continue
		chrom,position1,position2,ref,alt,qual,group_test_pvalue,depth_ref,depth_alt,depth_ref_samples,depth_alt_samples,genotype,other = line.rstrip("\n").split("\t")
		yield [chrom,position1,position2,ref,alt,qual,group_test_pvalue,depth_ref,depth_alt,depth_ref_samples,depth_alt_samples,genotype]
	f.close()

def merge_chromregion(regions):#["chrom",start,end] # start end is 0-based [  ) like bed
	mergedregion = []
	regions = utils.us_sort(regions,0,1,2) # must be sorted
	totallen = 0
	if len(regions) > 0:
		initchrom,initstart,initend = regions[0]
	else:
		return mergedregion,0
	for chrom,start,end in regions[1:]:
		if initchrom == chrom and start <= initend:
			initend = max(initend,end)
		else:
			totallen += initend - initstart
			mergedregion.append([initchrom,initstart,initend])
			initchrom = chrom
			initstart = start
			initend   = end
	mergedregion.append([initchrom,initstart,initend])
	totallen += initend - initstart
	return mergedregion,totallen

def merge_region(regions): ## must be sorted
	# regions = [[1,1000],[999,1100],[2000,3000]]
	# return [[1,1100],[2000,3000]]
	# regions must be sorted and   0-base [,)
	mergedregion = []
	if len(regions) > 0:
		initstart,initend = regions[0]	
	else: return mergedregion
	for start,end in regions[1:]:
		if start <= initend:
			initend = max(initend,end)
		else:
			mergedregion.append([initstart,initend])
			initstart = start
			initend = end
	mergedregion.append([initstart,initend])
	return mergedregion

def arf_read(arffn):
	f = compress.gz_file(arffn,"r")
	for line in f:
		if line.startswith("#"):continue
		rname,rleng,rstart,rend,rseq,gname,gleng,gstart,gend,gseq,gstrand,nmismatch,mathclabel = line.rstrip("\n").split("\t")
		yield [rname,rleng,rstart,rend,rseq,gname,gleng,gstart,gend,gseq,gstrand,nmismatch,mathclabel]
	f.close()

def fileread(fn):
	if not os.path.isfile(fn):
		sys.stderr.write("[Error] '%s' is not a file\n"%fn)
		sys.exit(1)
	if fn.endswith(".gz"):
		f = compress.gz_file(fn,"r")
	elif fn.endswith(".bz2"):
		f = compress.bz2file(fn)
	else:
		f = open(fn,"r")
	return f

def bed6_parse(fn):
	"""
	for ret in bed6_parse(sys.argv[1]):
		chrom,start,end,name,score,strandother = ret
	"""
	f = compress.gz_file(fn,"r")
	for line in f:
		if line.startswith("#"):continue
		try:
			chrom,start,end,name,score,strandother = line.rstrip().split("\t",5)
		except:
			sys.stderr.write("[ERROR] parse failed: %s"%line)
		yield [chrom,int(start),int(end),name,score,strandother]
	f.close()

def gff3_parse(fn):
	f = compress.gz_file(fn,"r")
	for line in f:
		if line.startswith("#"):continue
		chrom,source,seqtype,start,end,score,strand,phase,attributes = line.rstrip("\n").split("\t")
		yield [chrom,source,seqtype,start,end,score,strand,phase,attributes]
	f.close()

def refgene_parse(fn):
	f = compress.gz_file(fn,"r")
	for line in f:
		if line.startswith("#"):continue
		num,nm_name,chrom,strand,exon_s,exon_e,cds_s,cds_e,exon_num,exonstarts,exonends,uniq_id,symbol, kown1, kown2, exon_status = line.rstrip().split("\t")
		yield [num,nm_name,chrom,strand,exon_s,exon_e,cds_s,cds_e,exon_num,exonstarts,exonends,uniq_id,symbol, kown1, kown2, exon_status]
	f.close()

def miRNA_target_parse(fn):
	f = compress.gz_file(fn,"r")
	for line in f:
		if line.startswith("#"):continue
		arr = line.rstrip("\n").split("\t")
		microRNAid,detalmciroRNA,target_Genes = arr[0:3]
		UTR = arr[-3]
		pairing = arr[-2]
		miseq = arr[-1]
		yield [microRNAid,detalmciroRNA,target_Genes,UTR,pairing,miseq]
	f.close()


def sigfile_parse(fn):
	f = compress.gz_file(fn,"r")
	for line in f:
		if line.startswith("#"):continue
		anno1,anno2,fc,rawp,fdr = line.rstrip("\n").split("\t")
		yield [anno1,anno2,fc,rawp,fdr]
	f.close()

def soap_aln_parse(fn):
	f = compress.gz_file(fn,"r")
	for line in f:
		if line.startswith("#"):continue
		seqid,seqread,qual,mcounts,PEtag,length,strand,chrom,sitestart1,mismatch,cigar,match = line.rstrip("\n").split("\t")
		yield [seqid,seqread,qual,mcounts,PEtag,length,strand,chrom,sitestart1,mismatch,cigar,match]
	f.close()

def fasta_read(fn):
	"""
	seq.id -> ID: gi|2765658|emb|Z78533.1|CIZ78533
	seq.name  -> Name: gi|2765658|emb|Z78533.1|CIZ78533
	seq.description -> Description: gi|2765658|emb|Z78533.1|CIZ78533 C.irapeanum 5.8S rRNA gene and ITS1 and ITS2 DNA
	Number of features: 0
	seq.seq ->  Seq('CGTAACAAGGTTTCCGTAGGTGAACCTGCGGAAGGATCATTGATGAGACCGTGG...GGG', SingleLetterAlphabet())
	str(seq.seq)
	"""
	f = fileread(fn)
	for seq in SeqIO.parse(f,"fasta"):
		yield seq
	f.close()

def fastq_read(fn,qual):
	if qual not in ['fastq-sanger','fastq-solexa','fastq-illumina']:
		sys.stderr.write("[ERROR] Unknown quality\n")
		sys.exit(1)
	
	f = fileread(fn)
	"""
	rec.seq
	rec.letter_annotations['phred_quality']
	"""
	for seq in SeqIO.parse(f,qual):
		yield seq
	f.close()

def bwt_parse(fn):
	f = compress.gz_file(fn,"r")
	for line in f:
		query_id,strand,subject_id,pos,seq,qual,score,mismatch = line.rstrip("\n").split("\t")
		yield [query_id,strand,subject_id,pos,seq,qual,score,mismatch]
	f.close()

def blast6_parse(fn):
	f = compress.gz_file(fn,"r")
	for line in f:
		if line.startswith("#"):continue
		try:
			query_id, subject_id, identity, alignment_length, mismatches, gap_opens, qstart, qend, sstart, send, evalue, bitscore = line.rstrip("\n").split("\t")
		except:
			sys.stderr.write("[WARN] blast can not parse '%s'"%line)
			continue
		yield [query_id, subject_id, identity, alignment_length, mismatches, gap_opens, qstart, qend, sstart, send, evalue, bitscore]
	f.close()

def blast6v2_parse(fn):
	f = compress.gz_file(fn,"r")
	for line in f:
		if line.startswith("#"):continue
		try:
			qseqid,sseqid,pident,length,mismatch,gapopen,qstart,qend,sstart,send,evalue,stitle,sstrand,qcovs,qcovhsp = line.rstrip("\n").split("\t")
		except:
			sys.stderr.write("[WARN] blast can not parse '%s'"%line)
			continue
		yield [qseqid,sseqid,pident,length,mismatch,gapopen,qstart,qend,sstart,send,evalue,stitle,sstrand,qcovs,qcovhsp]
	f.close()

def gtf_parse(fn,add="chr"):
	f = compress.gz_file(fn,"r")
	for line in f:
		if line.startswith("#"):continue
		chrom,rnatype,region_type,start,end,score,strand,codon,commnet = line.rstrip("\n").split("\t")
		yield [add+chrom.lstrip("chr"),rnatype,region_type,start,end,score,strand,codon,commnet]
	f.close()

## def blast_parse
## def sam or bam parse ...
##

def sortmergebedfile(bedfile):
	hgenesregion = {} # gene: chrom,start, end  ## bedfile  0-based [ ) 
	bedfilef = open(bedfile,"r")
	for line in bedfilef:
		if line.startswith("#"):continue
		chrom,start,end,genename = line.rstrip("\n").split("\t")
		if genename not in hgenesregion: hgenesregion[genename] = []
		hgenesregion[genename].append([chrom,int(start),int(end)])
	bedfilef.close()
	for genename in hgenesregion:
		sortedregion,tmp = merge_chromregion(hgenesregion[genename])
		hgenesregion[genename] = sortedregion[:]
	return hgenesregion



if __name__ == "__main__":
	a = [[1,10],[17,22],[40,44],[42,47],[46,100],[101,408]]
	print(a)
	print(merge_region(a))



