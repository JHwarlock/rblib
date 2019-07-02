import sys
import numpy as np

## genelist = ["A","B","C","D","E"] ...
## geneA mutation in samples : ["S1","S2","S3"] , geneB mutation in samples = ["S2","S3","S4"]
## Xor + And genes 
## how to predict ? --> permutation test

## 
from rblib import gwaspls
import itertools

# ranodm 1 time, 
def randomMut(list1):
	lenlist = len(list1)
	idx = gwaspls.resampling2([1,]*lenlist,lenlist)
	return list1[idx].tolist()
#for i in xrange(10):
#	print randomMut([1,1,1,1,0,0,0])

def get_And_Xor(set1,set2,ntimes=1000):
	# use permutation test to do MRT
	setand = np.sum(set1 * set2)
	setxor = np.sum((set1 + set2)==1)
	print set1
	print set2
	print setand
	print setxor
	print len(set1)
	outand = []
	outxor = []
	count = 0
	sns = len(set1)
	#set1list = set1.tolist() # use MentoCarlo permutation 
	
	while count < ntimes:
	#for i in itertools.permutations(set1list, sns):
		set1mut = randomMut(set1)
		tmp1 = np.sum(np.asarray(set1mut) * set2)
		tmp2 = np.sum((np.asarray(set1mut) + set2)==1) 
		outand.append( tmp1 )
		outxor.append( tmp2 )
		print tmp1,tmp2
		count += 1
		if count > ntimes:break
		if count %10 == 0: sys.stderr.write("%d\n"%count)
	return np.sum( np.asarray(outand) >= setand) * 1.0/count,np.sum( np.asarray(outxor) >= setxor) * 1.0/count

if __name__ == "__main__":
	setA = set(["A","B","C"])
	setB  = set(["D","E","F","M","N"])
	setA = np.asarray([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	setB = np.asarray([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	print get_And_Xor(setA,setB,ntimes=10000)

#def bedregion_cal(bedfile,mutations,)

