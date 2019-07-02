import sys

def callrsqure():#reffreq,altfreq,refgenofreq,hetfreq,altgenofreq):
	pA = 0.3
	pB = 0.2
	pa = 0.7
	pb = 0.8
	pAB = pA * pB # x11
	paB = pa * pB # x21
	pAb = pA * pb # x12
	pab = pa * pb # x22
	D = pAB * pab - paB*pAb
	print D
callrsqure()

