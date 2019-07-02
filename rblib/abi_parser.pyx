import sys
import numpy as np
from Bio import SeqIO
from collections import defaultdict
from rblib import statplot

class ABIparse(object):
	def __init__(self):
		self.channels = ['DATA9', 'DATA10', 'DATA11', 'DATA12']
		self.record   = None
		self.seq1     = None
		self.seq2     = None
		self.seqlen   = 0
		self.trace    = defaultdict(list)
	def parse(self,fn):
		self.record   = SeqIO.read(fn, 'abi')
		self.seq1     = self.record.annotations['abif_raw']['PBAS1']
		self.seq2     = self.record.annotations['abif_raw']['PBAS2']
		self.seqlen   = len(self.seq1)
		assert self.seqlen == len(self.seq2)
		for c in self.channels:
			self.trace[c] = self.record.annotations['abif_raw'][c]
		return 0
	def plot(self,figprefix,figsize=(30,4),dpi=300,fontsize=4,seqpos=None,seqoffset=20): # focus 1 pos , 
		Garr = np.asarray(self.trace['DATA9'])
		Aarr = np.asarray(self.trace['DATA10'])
		Tarr = np.asarray(self.trace['DATA11'])
		Carr = np.asarray(self.trace['DATA12'])


		fig = statplot.plt.figure(dpi=dpi,figsize=figsize)
		ax = fig.add_subplot(111)
		#Aarr[np.asarray(record.annotations['abif_raw']['PLOC1'])[3]] = 1000
		ax.plot(Garr, color='blue')    ## "G"
		ax.plot(Aarr, color='red')     ## "A"
		ax.plot(Tarr, color='green')   ## "T"
		ax.plot(Carr, color='#CEBE29')  ## "C"
		xpos    = np.asarray(self.record.annotations['abif_raw']['PLOC1'])
		offset  = np.max(Garr + Aarr + Tarr + Carr) * 0.005
		ypos    = np.ones(self.seqlen) * -1 * offset
		for i in xrange(self.seqlen):
			ax.text(xpos[i],ypos[i],self.seq1[i],ha='center',va='top',fontsize=fontsize)
		
		if seqpos is None:
			xmin = xpos[0]-10
			xmax = xpos[-1] + 10
		else:
			xmin = xpos[seqpos-seqoffset] -10
			xmax = xpos[seqpos+seqoffset] -10
		ax.set_xlim(xmin,xmax)
		fig.savefig("%s.png"%figprefix)
		fig.savefig("%s.svg"%figprefix)
		statplot.plt.clf(); statplot.plt.close()
		return 0
	def save_to_fa(self,fn_out):
		fout = file(fn_out+".fa","w")
		fout.write(">seq1\n%s\n>seq2\n%s\n"%(self.seq1,self.seq2))
		fout.close()
		return 0

		
def abiparse_script(fn):
	abi_ins = ABIparse() 
	abi_ins.parse(fn)
	abi_ins.plot(fn+".abiplot")
	abi_ins.save_to_fa(fn+".abi_fa")
	return 0

if __name__ == "__main__":
	abiparse_script(sys.argv[1])



