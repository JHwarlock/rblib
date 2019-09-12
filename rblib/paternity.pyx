import sys
import numpy as np


def ssr_poly(n): # mutil-alleles # allele is 0,1,2,3,4    genotype =  (0,0) (1,2) ...
	alleles = list(range(n))
	prob = np.random.random(n)
	prob = prob/np.sum(prob)
	cumprob = np.cumsum(prob)

	return alleles,prob,cumprob

def generate_individual_geno(alleles,cumprob,ploidy=2):
	geno = []
	for i in range(ploidy):
		idx = np.sum(np.random.random() > cumprob)
		geno.append(alleles[idx])
	return "|".join(map(str,geno))

def population_geno_persite(snnum,alleles,cumprob,ploidy=2):
	popgeno = []
	for i in range(snnum):
		popgeno.append(generate_individual_geno(alleles,cumprob,ploidy))
	return popgeno


def population_genos(numbersites,snnum,ploidy=2,ssr=4):
	numsites = []
	probs = []
	for i in range(numbersites):
		alleles,prob,cumprob = ssr_poly(ssr)
		popgeno = population_geno_persite(snnum,alleles,cumprob,ploidy)
		numsites.append(popgeno)
		probs.append(prob)
	numsites_samples = np.asarray(numsites)
	return numsites_samples,probs
#print population_genos(6,4) # 

def generate_twoidx(snnum):
	i = 0; j=0
	while i==j:
		i,j = np.int32(np.random.random(2)*snnum)
	return i,j
	
def compare_ind(parent_vec,child_vec,prob,cutCPI=10000,cutCPE=0.9999):
	leng = len(parent_vec)
	assert leng == len(child_vec)
	nosame_num = 0
	counts = 0
	probs = []
	pis = []
	for i in range(leng):
		geno_parent = set(parent_vec[i].split("|"))
		geno_child  = set(child_vec[i].split("|"))
		#if "-" in geno_parent or "-" in geno_child:continue
		
		if len(geno_parent & geno_child) <= 0:
			nosame_num +=1
			if nosame_num > 3:break
			continue
		else:
			probs.append(prob[i])
			pi = calPI(geno_parent,geno_child,prob[i])
			pis.append(pi)
	if nosame_num > 3: return 0
	else:
		if np.prod(pis) > cutCPI and combine_PE(probs) > cutCPE:
			return 1
		else:
			return 0

## === script to get a population ====

def make_parents(numbersites = 300,snnum = 200, ploidy=2,ssr=4):
	population,probs = population_genos(numbersites,snnum,ploidy,ssr) # numbersites X snnum 
	return population,probs

### === to get generate ===
### random 2 sample, not the same
def make_childs(parents,num_childs = 500):
	records = []
	childs = []
	num_sites,snnum = parents.shape
	for i in range(num_childs):
		p1,p2 = generate_twoidx(snnum)
		records.append(set([p1,p2]))
		tmpchild = []
		for j in range(num_sites):
			## random to select, no linked
			geno1 = parents[j,p1].split("|")
			geno2 = parents[j,p2].split("|")
		
			allele1 = geno1[np.random.random()<0.5]
			allele2 = geno2[np.random.random()<0.5]
			geno_child = allele1 + "|" +  allele2
			tmpchild.append(geno_child)
		childs.append(tmpchild)
	childs = np.transpose(np.asarray(childs))
	return childs,records

def PE_index(parr):# for one site
	parr = np.asarray(parr)
	P = 1-4*np.sum(parr**2) + 2 * (np.sum(parr**2)**2) + 4*np.sum(parr**3) - 3 * np.sum(parr**4)
	return P # combine P : P = 1 - (1-P1)(1-P2)(1-P3)...(1-Pk)

def combine_PE(probs):
	parr = []
	for prob in probs:
		parr.append(PE_index(prob))
	return 1-np.prod(1-np.asarray(parr))

def calPI(setgeno1,setgeno2,prob): # parent ,child 
	crossgeone = setgeno1 & setgeno2
	crossgeone = map(int,list(crossgeone))
	if len(crossgeone) == 2:
		p = prob[crossgeone[0]]
		q = prob[crossgeone[1]]
		pi = (p+q)/(4*p*q)
	else:
		#print crossgeone
		assert len(crossgeone) == 1
		p = prob[crossgeone[0]]
		coef = len(setgeno2) * len(setgeno1) 
		pi = 1/(coef*p)
	return pi

## 2 kinds of relation ship
## 1st. parent to child,  relationship 
## 2nd. parent to no-child , no-relationship

##             False, True
##real False   
##real True
def get_relationship(parents,childs,records,probs,cutCPE=0.9999,cutCPI=10000):
	nsites1,numparents = parents.shape
	nsites2,numchilds  = childs.shape
	assert nsites1 == nsites2
	current = np.zeros((2,2))
	count = 0
	for i in range(numparents):
		parent1 = parents[:,i]
		for j in range(numchilds):
			child1 = childs[:,j]
			yesorno = compare_ind(parent1,child1,probs,cutCPI=cutCPI,cutCPE=cutCPE)
			current[int(i in records[j]),yesorno] += 1
			count += 1
			if count % 10000 == 0:
				sys.stderr.write("[INFO] %d compares\n"%count)
	return current

### https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

def TPcall(arr1,arr2):
	# arr1 = np [1,1,1,0,0,0] 1 is true 0 is false
	# arr2 = np [1,0,1,1,0,1 ] 1 is predict true 0 is predict false
	TP = np.sum((arr1 + arr2) == 2)
	TN = np.sum((arr1 + arr2) == 0)
	FP = np.sum(arr1[(arr1 + arr2) == 1] == 0)
	FN = np.sum(arr1[(arr1 + arr2) == 1] == 1)
	return map(float,[TP,TN,FP,FN])

def classmeasure(TP,TN,FP,FN):
	# sensitivity, recall, hit rate, or true positive rate (TPR)
	TPR = TP*1.0/(TP+FN)
	# specificity or true negative rate (TNR)
	TNR = TN*1.0/(TN+FP)
	# precision or positive predictive value (PPV)
	PPV = TP*1.0/(TP+FP)
	# negative predictive value (NPV)
	NPV = TN*1.0/(TN+FN)
	# miss rate or false negative rate (FNR)
	FNR = 1-TPR # FN/(FN+TP)
	# fall-out or false positive rate (FPR)
	FPR = 1-TNR # FP/(FP+TN)
	# false discovery rate (FDR)
	FDR = 1-PPV # FP/FP+TP
	# false omission rate (FOR)
	FOR = 1-NPV
	# accuracy (ACC)
	ACC = (TP+TN)*1.0/(FN+TP+FP+TN) # (TP+TN)/(P+N)
	# F1 score is the harmonic mean of precision and sensitivity
	F1 = 2*(PPV*TPR) *1.0/ (PPV+TPR) # 2TP/(2TP+FP+FN)
	# Matthews correlation coefficient (MCC)
	MCC = (TP*TN-FP*FN)*1.0/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)*1.0)
	# Informedness or Bookmaker Informedness (BM)
	BM = TPR + TNR - 1
	# Markedness (MK)
	MK = PPV + NPV - 1
	return TPR, TNR, PPV,NPV,FNR,FPR,FDR,FOR,ACC,F1,MCC,BM,MK
"""
num_sites = [10,20,30,40,50]
PEx = []
ssrtypes = [2,3,4,5,6]

for num_sitesx in num_sites:
	for ssrtype in ssrtypes:
		PEtmp = []
		for i in range(500):
			parents,probs = make_parents(numbersites = num_sitesx,snnum = 2,ssr=ssrtype)
			PEtmp.append(combine_PE(probs))
		print str(num_sitesx) + "\t" + str(ssrtype) + "\t" + "\t".join(map(str,PEtmp))





#exit(0)
num_sites = [20,30,40,50,60]

snnum = 300

out1 = 
#number of sites     PE
       %d            %.4f

out2 =        predict False, predict True
real False          %d             %d 
real True           %d             %d

ssrtype = 7
for num_sitesx in num_sites:
	parents,probs = make_parents(numbersites = num_sitesx,snnum = snnum,ssr=ssrtype)
	CPE = combine_PE(probs)
	print out1%(num_sitesx,CPE)
	childs,records = make_childs(parents,num_childs = 300)
	#print records 
	result =  get_relationship(parents,childs,records,probs)#,cutCPE=combine_PE(probs)-0.004,cutCPI=7500)
	print out2%(result[0,0],result[0,1],result[1,0],result[1,1])
	acc = classmeasure(result[1,1],result[0,0],result[0,1],result[1,0])
	print "\t".join(map(str,acc))
	print "|".join(map(str,[ssrtype,num_sitesx,CPE]))+"|" + "|".join(map(str,np.asarray(acc)[[0,1,2,6,8]].tolist()))
"""
