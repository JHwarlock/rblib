import sys

"""
# to process genestruct's sequence and coords transfer
## here, we need to process 3 kinds of transfer

1. 
genome coords => exon position or cds  position     
			  => protein position  
2. 
cds or exon position => genome coords

3. 
protein  => cds
"""

def genome2exon(genomeleft,genomeright,region):# to exon or cds, a genomepos list
	## ACTC - , is a InDel,  ACT is in the cds region, but C not in it, 0-based [a,b)
	transleft,transright = 0,0
	assert genomeleft <> genomeright
	#regionlength = 0
	for start,end in region:
		#regionlength += (end-start)
		if genomeright > start and genomeleft < end:
			tmpgenomeleft = max(genomeleft,start)
			tmpgenomeright = min(genomeright,end)
			transleft += tmpgenomeleft - start
			transright += tmpgenomeright - start  # ? genomeleft
		elif genomeleft >= end:
			transleft += end - start
			transright += end - start
		else:
			break # not break ,because of to get whole region length for reverse 
	if transleft <> transright:
		return transleft,transright #,regionlength
	else:
		return None

def reverse_mRNApos(genomeleft,genomeright,regionlength):
	return regionlength - genomeright, regionlength - genomeleft ## transferred also 0-based [a,b)


def exon2genome(transleft,transright,region): ## 0-based [a,b)
	tmplen = 0
	for start,end in region:
		transleft -= tmplen
		transright-= tmplen
		tmplen = end-start
		if 0 <= transleft < tmplen:
			genomeleft = start + transleft
		if  0 < transright <= tmplen:
			genomeright = start + transright
		elif transright <= 0:
			break
	return genomeleft,genomeright

# when we need protein pos,  divmod(pos,3)

if __name__  == "__main__":
	#cds = [[7,76],[172,189],[199,230],[267,300]]
	#genomeleft  = 173
	#genomeright = 220
	#transleft,transright =  genome2exon(genomeleft,genomeright,cds)
	#print transleft,transright
	#print exon2genome(transleft,transright,cds)	
	"10043710        10044114"	
	cds = []
	f = file(sys.argv[1],"r")
	for line in f:
		cds.append(line.rstrip("\n").split("\t"))
	genomeleft = 10043710
	genomeright = 10044114
	transleft,transright =  genome2exon(genomeleft,genomeright,cds)
	print transleft,transright 
