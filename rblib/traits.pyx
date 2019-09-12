import sys
import pandas as pd
import numpy as np
class TraitInfo(object):
	def __init__(self):
		self.traitnum = 0
		self.snnum    = 0
		self.sns      = []
		self.dframe   = None
		self.traittype= {} ## string for Classification variables,  float for Continuous variables, use "trait:category" "trait:quantificat"
		self.traitname= []
		#self.traitlevel= {}
		self.traitidx = {}
		self.defaulttype = {
				"category"    : str,
				"quantificat" : float,
				}
	def __str__(self):
		strout  = "===========================================\n"
		strout += "Trait numbers  : %d\n"%self.traitnum
		strout += "Sample numbers : %d\n"%self.snnum
		strout += "Samples names  : %s\n"%(",".join(self.sns))
		strout += "-----------------------------------\n"
		strout += "Data:\n%s\n"%self.dframe.head(3)
		strout += "-----------------------------------\n"
		strout += "Traitname      : %s\n"%(",".join(self.traitname))
		#strout += "%s\n"%str(self.traitlevel)
		strout += "%s\n"%str(self.traitidx)
		strout += "===========================================\n"
		return strout
	def parse(self,traitinfo):
		f = open(traitinfo,"r")
		data = []
		for line in f:
			if line.startswith("###"):
				arr = line.strip().strip("###").split(":")
				assert arr[1] in self.defaulttype
				self.traittype[arr[0]] = arr[1]
				continue

			if line.startswith("##"):continue # which is header
			if line.startswith("#"):
				self.traitname = line.strip()[1:].split("\t") # columns include sample names
				try:
					assert sorted(self.traittype.keys()) == sorted(self.traitname)
				except:
					sys.stderr.write("[ERROR] assert sorted(self.traittype.keys()) == sorted(self.traitname)\n")
					sys.stderr.write("%s\n"%str(sorted(self.traittype.keys())))
					sys.stderr.write("%s\n"%str(sorted(self.traitname)))
					sys.stderr.write("line: %s"%line)
					sys.exit(1)
				self.traitnum = len(self.traitname)
				for i in range(len(self.traitname)):
					#self.traitlevel[self.traitname[i]] = []
					self.traitidx[i] = self.traitname[i]
				continue
			else:
				arr = line.strip().split("\t")
				self.snnum += 1
				self.sns.append(arr[0])
				data.append(arr)
		self.dframe = pd.DataFrame(np.asarray(data),index=self.sns,columns=self.traitname)
		for traitname in self.traittype:
			if self.traittype[traitname] == "category":
				self.dframe[traitname] = self.dframe[traitname].astype("category")
			elif self.traittype[traitname] == "quantificat":
				self.dframe[traitname] = np.float64(self.dframe[traitname])
		#sys.stderr.write("[INFO] Data read: \n%s\n[INFO] Done!\n"%str(self.dframe))
		return 0

if __name__ == "__main__":
	ti = TraitInfo()
	ti.parse(sys.argv[1])
	print(ti)

