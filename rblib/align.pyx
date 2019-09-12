import sys
from Bio import Align

def _calculate_identity(best_alignments):
	seqA,match,seqB = str(best_alignments).rstrip().split("\n")
	assert len(seqA) == len(seqB)
	count = 0
	totlen = 0
	for i in range(len(match)):
		if match[i] == ".":continue
		totlen += 1
		if seqA[i] == seqB[i] and seqA[i] != "-":
			count += 1
	return count,totlen


def simalign(seqA,seqB):
	aligner = Align.PairwiseAligner()
	#print aligner.mode
	#print aligner.algorithm
	aligner.mode = "local"
	#print aligner.algorithm
	#print(aligner)
	alignments = aligner.align(seqA.upper(), seqB.upper())
	bestaln = alignments[0]
	count,totlen = _calculate_identity(bestaln)
	#print bestaln,alignments.score
	iden = count *1.0  / max(min(len(seqA),len(seqB)),totlen) * 100
	return iden

def runtest():
	try:
		print(simalign("GACGTAcggg","AAGACGATACCCGGG"))
		return 0
	except Exception as error:
		print(error)
		return 1
if __name__ == "__main__":
	runtest()




