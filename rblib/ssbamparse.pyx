# -*- coding: UTF-8 -*-
import sys
import commands
import bamio
import pysam
# 4中 R，F , RF, FR
import time


def parse_unstrand_bam(fbamfile1,fbamfile2,chrom,start,end,sslib=0):
	"""
	fbamfile1 和 fbamfile2 为同1个文件，用pysam.Samfile 打开2次
	unstrand lib: for SE PE  
	sslib = -1 即为 SE 数据 非链特异
	sslib = 0  即为 PE 数据 非链特异
	"""
	#samfile = pysam.Samfile(bamfile, "rb" )
	#samfile2 = pysam.Samfile(bamfile,"rb")
	if sslib == 0:
		it = fbamfile1.fetch(chrom,start,end)	
		for read in it:
			if read.is_unmapped:continue
			if read.is_secondary:continue
			if not read.is_paired:continue
			if read.is_unmapped or read.mate_is_unmapped: continue
			if read.is_read1:
				if read.reference_id != read.next_reference_id:
					sys.stderr.write("[WARN] PE on different chromsome\nread1-> %s\nread2-> %s\n"%(str(read),str(read2)))
					continue
				read2 = fbamfile2.mate(read) 
				read1blocks =  read.get_blocks()
				read2blocks = read2.get_blocks()
				if read.is_reverse:
					inter = [read2blocks[-1][-1],read1blocks[0][0]]
					strand = "."
					if read.template_length >=0:
						sys.stderr.write("[WARN] PE aligned confused:\nread1-> %s\nread2-> %s\n"%(str(read),str(read2)))
						continue
				else:
					strand = "."
					inter = [read1blocks[-1][-1],read2blocks[0][0]]
					if read.template_length <=0:
						sys.stderr.write("[WARN] PE aligned confused:\nread1-> %s\nread2-> %s\n"%(str(read),str(read2)))
						continue
				#print read1blocks
				#print read2blocks
				#print inter
				#print strand
				yield [read1blocks,read2blocks,inter,strand]
	else:
		it = fbamfile1.fetch(chrom,start,end)
		for read in it:
			if read.is_unmapped:continue
			if read.is_secondary:continue
			read1blocks =  read.get_blocks()
			strand = "."
			yield [read1blocks,None,None,strand]

hstrand = {"+":0,".":0,"-":1}
def getcovpos(chrom,start,end,fbamfile1,fbamfile2,sslib):# 可以采用多种计数方式,这里默认采用union 的方式。
	# return scaled value
	if sslib >= 1:
		parsefun = parse_strand_specificbam
	else:
		parsefun = parse_unstrand_bam
	count_plus,count_minus,tpm_plus,tpm_minus = 0,0,0,0 
	counts = [count_plus,count_minus,tpm_plus,tpm_minus]
	#print chrom,start,end
	for ret in parsefun(fbamfile1,fbamfile2,chrom,start,end,sslib):
		read1blocks,read2blocks,inter,strand_t  = ret
		#allblocks = [] 
		if read1blocks or read2blocks:
			#allblocks += read1blocks
			counts[hstrand[strand_t]] += 1
		#if read2blocks:
			#allblocks += read2blocks
		#	counts[hstrand[strand_t]] += 1
		#for start_t,end_t in allblocks:
		#	if end > start_t and start < end_t:
		#		counts[hstrand[strand_t]] += 1
	counts[2] = float(counts[0]) / fbamfile1.mapped * 1000000/10*1000
	counts[3] = float(counts[1]) / fbamfile1.mapped * 1000000/10*1000
	#print counts
	return counts

def parse_strand_specificbam(fbamfile1,fbamfile2,chrom,start,end,sslib=3):
	"""
	fbamfile1 和 fbamfile2 为同1个文件，用pysam.Samfile 打开2次
	R: the single read is in the antisense (reverse) orientation
	F: the single read is in the sense (forward) orientation
	RF: first read (/1) of fragment pair is sequenced as anti-sense (reverse(R)), and second read (/2) is in the sense strand (forward(F)); typical of the dUTP/UDG sequencing method. (dUTP)
	FR:  first read (/1) of fragment pair is sequenced as sense (forward), and second read (/2) is in the antisense strand (reverse)
	unstrand:
	"""
	spe = 0
	#it = samfile.fetch(chrom,start,end)
	if sslib >2:spe = 1 #双端数据
	if sslib == 1:
		it = fbamfile1.fetch(chrom,start,end)
		for read in it:
			if read.is_unmapped:continue
			if read.is_secondary:continue
			read1blocks =  read.get_blocks()
			if read.is_reverse:
				strand = "+"
			else:strand = "-"
			yield [read1blocks,None,None,strand]
	elif sslib == 2:
		it = fbamfile1.fetch(chrom,start,end)
		for read in it:
			if read.is_unmapped:continue
			if read.is_secondary:continue
			read1blocks =  read.get_blocks()
			if read.is_reverse:
				strand = "-"
			else:strand = "+"
			yield [read1blocks,None,None,strand]
	#count = 0
	elif sslib == 3:
		"PE 数据必须2个read 都有mapping"
		it = fbamfile1.fetch(chrom,start,end)
		for read in it:
			if read.is_unmapped:continue
			if read.is_secondary:continue
			if not read.is_paired:continue
			if read.is_unmapped or read.mate_is_unmapped: continue
			if read.is_read1:
				#count += 1
				#print read
				#t1 = time.time()
				if read.reference_id != read.next_reference_id:
					sys.stderr.write("[WARN] PE on different chromsome\nread1-> %s\nread2-> %s\n"%(str(read),str(read2)))
					continue
				read2 = fbamfile2.mate(read)
				#print samfile.references
				#print time.time()-t1
				#print read2
				read1blocks =  read.get_blocks()
				read2blocks = read2.get_blocks()
				if read.is_reverse:
					#read1blocks =  read.get_blocks()#[(3564, 3616), (4356, 4380)]   ## template_length
					#read2blocks = read2.get_blocks()# 
					# 注意， 此处需避免比对到不同的染色体or间距过长的问题，因为2个exon 间若距离过长，则很可能是融合基因。
					# 融合基因需要另外再处理。
					# 融合基因在tophat - fusion 中的tag 标记是 XF and XP
					# 目前此程序仅不针对融合基因处理，融合基因需用另外的处理方式
					inter = [read2blocks[-1][-1],read1blocks[0][0]]
					strand = "+"
					if read.template_length >=0:
						sys.stderr.write("[WARN] PE aligned confused:\nread1-> %s\nread2-> %s\n"%(str(read),str(read2)))
						continue
				else:
					strand = "-"
					#read1blocks = read.get_blocks()
					#read2blocks = read2.get_blocks()
					inter = [read1blocks[-1][-1],read2blocks[0][0]]
					if read.template_length <=0:
						sys.stderr.write("[WARN] PE aligned confused:\nread1-> %s\nread2-> %s\n"%(str(read),str(read2)))
						continue
				## strand 记录了比对到基因组+链  和 - 链上的信息。
				yield [read1blocks,read2blocks,inter,strand]
	elif sslib == 4:
		it = fbamfile1.fetch(chrom,start,end)
		for read in it:
			if read.is_unmapped:continue
			if read.is_secondary:continue
			if not read.is_paired:continue
			if read.is_unmapped or read.mate_is_unmapped: continue
			if read.is_read1:
				if read.reference_id != read.next_reference_id:
					sys.stderr.write("[WARN] PE on different chromsome\n%s\n%s\n"%(fbamfile1.references[read.reference_id],fbamfile1.references[read.next_reference_id]))
				read2 = fbamfile2.mate(read)
				read1blocks =  read.get_blocks()
				read2blocks = read2.get_blocks()
				if read.is_reverse:
					strand = "-"
					inter = [read2blocks[-1][-1],read1blocks[0][0]]
					if read.template_length >=0:
						sys.stderr.write("[WARN] PE aligned confused:\nread1-> %s\nread2-> %s\n"%(str(read),str(read2)))
						continue
				else:
					strand = "+"
					inter = [read1blocks[-1][-1],read2blocks[0][0]]
					if read.template_length <=0:
						sys.stderr.write("[WARN] PE aligned confused:\nread1-> %s\nread2-> %s\n"%(str(read),str(read2)))
						continue
				yield [read1blocks,read2blocks,inter,strand]	
	#return 0

if __name__ == "__main__":
	for ret in parse_strand_specificbam(sys.argv[1],"Chr1",2000,4000,sslib=4):
		print(ret)

