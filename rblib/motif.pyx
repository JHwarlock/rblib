import sys
from rklib import compress
from rklib import utils
from Bio import motifs
import subprocess
try:
	from Bio.motifs.jaspar.db import JASPAR5
except:
	sys.stderr.write("[WARN] Bio.motifs.jaspar.db import failed")
	flag = 1
	sys.exit(1)

class motif_jaspar(object):
	def __init__(self,dbinfo = None):
		"""
		#number accession       version name    class   family  medline protein taxid:species
		9342    MA0113  1       NR3C1   Zinc-coordinating       Hormone-nuclear Receptor        15563547        P04150  10090:Mus musculus// 10116:Rattus norvegicus
		uniq key is accession+"."+version
		"""
		self.db = "jaspar"
		self.motifs = []
		self.n = 0
		self.info = {}
		##sqlite3   sql   or  tsv shelve ## use the suffix: .sqlite3  .sql .txt .db
		if dbinfo is None:pass
		else:
			f = open(dbinfo,"r")
			for line in f:
				if line.startswith("#"):continue
				number,acc,version,name,mclass,family,medline,protein,taxid_s = line.rstrip("\n").split("\t")
				tmp = taxid_s.split("//")
				taxids = []
				for ts in tmp:
					if ts.startswith("-"):continue
					t,s = ts.rstrip().split(":")
					taxids.append(t)
				key = ".".join([acc,version])
				self.info[key.strip()] = [name.strip(),mclass.strip(),family.strip(),medline,protein,taxids]
			f.close()
	def parse_pfm(self,f):
		f = compress.gz_file(f,"r")
		for m in motifs.parse(f,"jaspar"):
			key = m.matrix_id.strip()
			if key in self.info:
				name,mclass,family,medline,protein,taxids = self.info[key]
				m.name = m.name.strip()
				m.tf_class = mclass
				m.tf_family = family
				m.medline = medline
				m.collection = protein
				m.tax_group = taxids[:]
			self.motifs.append(m)
		f.close()
		self.n = len(self.motifs)
		return 0
	def parse_sites(self,f):
		return 0
#====================================
"""
	def cal_pwm(self,normalize={'A':0.6, 'C': 0.4, 'G': 0.4, 'T': 0.6}):
		for m in self.motifs:
			m.pwm = m.counts.normalize(pseudocounts=normalize)
	def cal_pssm(self,)
"""
class motif(object):
	def __init__(self,motif_instance):
		self.m = motif_instance
		self.length = motif_instance.length
		self.alphabet = self.m.alphabet
		self.counts = motif_instance.counts
		self.background = motif_instance.background
		self.consensus = motif_instance.consensus
		#By default, the Perl TFBS modules use a pseudocount equal to
		#sqrt(N)*bg[nucleotide, = 0.25 uniform], where N represents
		#the total number of sequences used to construct the matrix.
		
		self.id = self.m.matrix_id# id is acc+"."+version  : MA0520.2,which is uniq key
		self.name  = self.m.name.strip()
		#self.acc = None
		self.tfclass = self.m.tf_class
		self.family = self.m.tf_family
		self.medline = self.m.medline
		self.protein = self.m.collection
		self.taxids = self.m.tax_group
		try:
			self.pseudocounts = motifs.jaspar.calculate_pseudocounts(self.m)
		except:
			self.pseudocounts = 0.5
		self.pwm = None
		self.pssm = None
		self.pssm_max = None
		self.pssm_min = None
		self.pssm_mean = None
		self.pssm_std = None
		self.rel_score_cutoff = 0.9
		self.fpr = 0.01
		self.fnr = 0.1
		self.fpr_fnr = 1000
		self.distribution = None
		self.threshold = None
	def calcore(self,pseudocounts = None,background = None):
		if pseudocounts == None:
			self.pwm = self.counts.normalize(pseudocounts=self.pseudocounts)
		else:
			self.pwm = self.counts.normalize(pseudocounts=pseudocounts)
		"""
		A uniform background is used if background is not specified. The mean is particularly important, as its
		value is equal to the Kullback-Leibler divergence or relative entropy, and is a measure for the information
		content of the motif compared to the background. As in Biopython the base-2 logarithm is used in the
		calculation of the log-odds scores, the information content has units of bits.
		"""
		if background == None:
			self.pssm = self.pwm.log_odds(self.background)
			self.pssm_mean = self.pssm.mean(self.background)
			self.pssm_std = self.pssm.std(self.background)
			self.distribution = self.pssm.distribution(background=self.background, precision=10**4)
		else:
			self.pssm = self.pwm.log_odds(background)
			self.pssm_mean = self.pssm.mean(background)
			self.pssm_std = self.pssm.std(background)
			self.distribution = self.pssm.distribution(background=background, precision=10**4)
		self.pssm_max = self.pssm.max
		self.pssm_min = self.pssm.min
	def cal_threshold(self,method="fpr"):
		if method == "fpr":
			self.threshold = self.distribution.threshold_fpr(self.fpr)
		elif method == "rel":
			self.threshold = (self.pssm_max - self.pssm_min)*self.rel_score_cutoff + self.pssm_min
		elif method == "fnr":
			self.threshold = self.distribution.threshold_fnr(self.fnr)
		elif method == "fpr_fnr":
			self.threshold = self.distribution.threshold_balanced(self.fpr_fnr)
		else:
			sys.stderr.write("[ERROR] Unkown method: %s"%method)
			return None
	def search(self,seq,cutoff=1.0):
		ret = []
		cuttmp = min(self.rel_score_cutoff,cutoff)
		for pos,score in self.pssm.search(seq,threshold = self.threshold):
			tmpseq = seq[pos:pos+self.length]
			rel_score = (score - self.pssm_min) / (self.pssm_max - self.pssm_min)
			if rel_score < cuttmp:continue
			ret.append([pos,score,rel_score,tmpseq])
		if ret:
			return ret
		else:
			return None
	def plot_motif(self,name = None,outdir="motif_plot"):
		utils.dirDetectCreate(outdir)
		if name == None:
			name = self.name
		"""
		try:
			self.m.weblogo(outdir+"/"+name+".png",stack_width="large",show_errorbars=False,color_scheme="color_classic",show_fineprint=False,logo_title=name)
			self.m.weblogo(outdir+"/"+name+".eps",show_errorbars=False,color_scheme="color_classic",format="eps",show_fineprint=False,logo_title=name)
		except:
			pass
		weblogo < test.matrix.txt --size large --title name --fineprint name  --errorbars NO -c classic   -D transfac > test.eps
		"""
		ftmp = open(outdir+"/"+".tmp.txt","w")
		ftmp.write(str(self.m.format("transfac")))
		ftmp.close()
		cmd1 = "weblogo -f %s --size large --scale-width NO --fineprint %s --errorbars NO -c classic -D transfac -F png --resolution 1000 1>%s.png"%(outdir+"/"+".tmp.txt",name,outdir+"/"+name)
		cmd2 = "weblogo -f %s --size large --scale-width NO --fineprint %s --errorbars NO -c classic -D transfac -F svg --resolution 1000  1>%s.svg"%(outdir+"/"+".tmp.txt",name,outdir+"/"+name)
		(exitstatus, outtext) = subprocess.getstatusoutput(cmd1)
		return 0

