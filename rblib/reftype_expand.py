import sys
from  Bio.Seq import Seq
from rblib import seqio,bamio,defs
import os
f = file(os.path.dirname(os.path.abspath(__file__))+os.path.sep+"ref.types","r")
for line in f:
	arr = line.rstrip("\n").split("\t")
	context = arr[0]
	ref = arr[1]
	alt = arr[2]
	idx = arr[3]

	key = "%s:%s>%s"%(arr[0],arr[1],arr[2])
	triseq = str(Seq(context).reverse_complement())
	rctriseq = str(Seq(triseq).reverse_complement())
	refrc = defs.hrc[ref]
	altrc = defs.hrc[alt]
	print "\t".join([context, triseq, rctriseq, ref, refrc, alt, altrc, idx, str(int(idx)+96)])
f.close()


