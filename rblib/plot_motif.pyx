import sys
import motif
import seqio
import enrich
from Bio.Seq import Seq
tfinfo = sys.argv[1]
tfpfm = sys.argv[2]
motif_db = motif.motif_jaspar(tfinfo)
motif_db.parse_pfm(tfpfm)

hinfo = {}
for motif_instance in motif_db.motifs:
	tmpmotif = motif.motif(motif_instance)
	flag = 0
	for taxid in tmpmotif.taxids:
		if taxid in ["9606"]:
			flag = 1
			break
	if not flag:continue
	hinfo[tmpmotif.name] = tmpmotif
enrich = file(sys.argv[3],"r")
for line in enrich:
	tf,other = line.rstrip("\n").split("\t",1)
	if tf in hinfo:
		tmpmotif = hinfo[tf]
		name = tmpmotif.name
		family = tmpmotif.family
		classnm = tmpmotif.tfclass
		protein = tmpmotif.protein
		tmpmotif.plot_motif()
		print "\t".join([tf,other,name,family,classnm,protein])
enrich.close()


