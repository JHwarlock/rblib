# -*- coding: UTF-8 -*-
# 简并碱基Degenerate base
hchr= {"chr1":"1",
		"chr2":"2",
		"chr3":"3",
		"chr4":"4",
		"chr5":"5",
		"chr6":"6",
		"chr7":"7",
		"chr8":"8",
		"chr9":"9",
		"chr10":"10",
		"chr11":"11",
		"chr12":"12",
		"chr13":"13",
		"chr14":"14",
		"chr15":"15",
		"chr16":"16",
		"chr17":"17",
		"chr18":"18",
		"chr19":"19",
		"chr20":"20",
		"chr21":"21",
		"chr22":"22",
		"chrX":"X",
		"chrY":"Y",
		"chrM":"MT",
		"chrMT":"MT",
		}

hg38chrlen = {
		"chr1":248956422,
		"chr2":242193529,
		"chr3":198295559,
		"chr4":190214555,
		"chr5":181538259,
		"chr6":170805979,
		"chr7":159345973,
		"chr8":145138636,
		"chr9":138394717,
		"chr10":133797422,
		"chr11":135086622,
		"chr12":133275309,
		"chr13":114364328,
		"chr14":107043718,
		"chr15":101991189,
		"chr16":90338345,
		"chr17":83257441,
		"chr18":80373285,
		"chr19":58617616,
		"chr20":64444167,
		"chr21":46709983,
		"chr22":50818468,
		"chrX":156040895,
		"chrY":57227415,
		"chrM":16569,
		"chrMT":16569,
		}

hdg = {"A":"A/A",
		"C":"C/C",
		"G":"G/G",
		"T":"T/T",
		"R":"A/G",
		"Y":"C/T",
		"M":"A/C",
		"K":"G/T",
		"S":"G/C",
		"W":"A/T",
		"H":"A/T/C",
		"B":"G/T/C",
		"V":"G/A/C",
		"D":"G/A/T",
		"N":"A/T/C/G",
		}
import re
nuclpt  = re.compile("[^ACGTNU]",re.I)
reppt   = re.compile("N|n|a|c|g|t") # or [Nnacgt]
gcconpt = re.compile("G|C",re.I)
cpg     = re.compile("CG",re.I)

hmutidx = {"AG":0,
		"TC":0, # AT_Transitions
		"AC":1, # AT_Transversions
		"AT":1,
		"TA":1,
		"TG":1,
		"CT":2, # CG_Transitions 
		"GA":2,
		"CA":3, # CG_Transversions
		"CG":3,
		"GC":3,
		"GT":3, 
		} 


# 1. 核苷酸互补表
hrc = {'A':'T',
		'a':'T',
		'T':'A',
		't':'A',
		'U':'A',
		'u':'A',
		'C':'G',
		'c':'G',
		'G':'C',
		'g':'C',
		'N':'N',
		'n':'N',
		}
hnuclidx = {'A':0,
		'a':0,
		'C':1,
		'c':1,
		'G':2,
		'g':2,
		'T':3,
		't':3,
		'U':3,
		'u':3,
		'N':4,
		'n':4,
		'CG':5,
		'cG':5,
		'cg':5,
		'Cg':5,
		}
hrfam = {
		"Cis-reg_leader":"Cis-reg",
		"Intron":"Intron",
		"Gene_antitoxin":"antitoxin",
		"Cis-reg":"Cis-reg",
		"Gene_tRNA":"tRNA",
		"Gene_snRNA_splicing":"snRNA",
		"Gene_miRNA":"miRNA",
		"Gene_sRNA":"sRNA",
		"Gene_lncRNA":"lncRNA",
		"Cis-reg_thermoregulator":"Cis-reg",
		"Gene_snRNA_snoRNA_HACA-box":"snoRNA",
		"Cis-reg_frameshift_element":"Cis-reg",
		"Gene_snRNA_snoRNA_scaRNA":"snoRNA",
		"Gene_CRISPR":"CRISPR",
		"Cis-reg_IRES":"Cis-reg",
		"Gene_antisense":"antisense",
		"Gene_ribozyme":"ribozyme",
		"Cis-reg_riboswitch":"Cis-reg",
		"Gene_snRNA_snoRNA_CD-box":"snoRNA",
		"Gene":"Gene",
		"Gene_rRNA":"rRNA",
		"Gene_snRNA":"snRNA",
		}

"""
['Cis-reg_leader', 'Intron', 'Gene_antitoxin', 'Cis-reg', 'Gene_tRNA', 'Gene_snRNA_splicing', 'Gene_miRNA', 'Gene_sRNA', 'Gene_lncRNA', 'Cis-reg_thermoregulator', 'Gene_snRNA_snoRNA_HACA-box', 'Cis-reg_frameshift_element', 'Gene_snRNA_snoRNA_scaRNA', 'Gene_CRISPR', 'Cis-reg_IRES', 'Gene_antisense', 'Gene_ribozyme', 'Cis-reg_riboswitch', 'Gene_snRNA_snoRNA_CD-box', 'Gene', 'Gene_rRNA', 'Gene_snRNA']
"""

hgeneclass = { 
        "3prime_overlapping_ncrna":"lnc",            # Long non-coding RNA genes
        "antisense":"lnc",                           # Long non-coding RNA genes
        "IG_C_gene":"imm",                           # Immunoglobulin/T-cell receptor gene segments     protein coding segments
        "IG_C_pseudogene":"imm",                     # Immunoglobulin/T-cell receptor gene segments     pseudogenes
        "IG_D_gene":"imm",                           # Immunoglobulin/T-cell receptor gene segments     protein coding segments
        "IG_J_gene":"imm",                           # Immunoglobulin/T-cell receptor gene segments     protein coding segments
        "IG_J_pseudogene":"imm",                     # Immunoglobulin/T-cell receptor gene segments     pseudogenes
        "IG_V_gene":"imm",                           # Immunoglobulin/T-cell receptor gene segments     protein coding segments
        "IG_V_pseudogene":"imm",                     # Immunoglobulin/T-cell receptor gene segments     pseudogenes
        "known_ncrna":"lnc",                         # Long non-coding RNA genes 
        "lincRNA":"lnc",                             # Long non-coding RNA genes 
        "miRNA":"srna",                              # Small non-coding RNA genes 
        "misc_RNA":"srna",                           # Small non-coding RNA genes 
        "Mt_rRNA":"srna",                            # Small non-coding RNA genes 
        "Mt_tRNA":"srna",                            # Small non-coding RNA genes 
        "non_coding":"lnc",                          # Long non-coding RNA genes 
        "polymorphic_pseudogene":"pseudo",           # Pseudogenes      polymorphic pseudogenes
        "processed_pseudogene":"pseudo",             # Pseudogenes      processed pseudogenes
        "processed_transcript":"lnc",                # Long non-coding RNA genes   
        "protein_coding":"coding",                   # Protein-coding genes
        "pseudogene":"pseudo",                       # Pseudogenes      pseudogenes
        "rRNA":"srna",                               # Small non-coding RNA genes
        "sense_intronic":"lnc",                      # Long non-coding RNA genes
        "sense_overlapping":"lnc",                   # Long non-coding RNA genes
        "snoRNA":"srna",                                         # Small non-coding RNA genes
        "snRNA":"srna",                                          # Small non-coding RNA genes
        "TEC":"lnc",                                             # Long non-coding RNA genes
        "transcribed_processed_pseudogene":"pseudo",             # Pseudogenes      processed pseudogenes
        "transcribed_unitary_pseudogene":"pseudo",               # Pseudogenes      unitary pseudogenes
        "transcribed_unprocessed_pseudogene":"pseudo",           # Pseudogenes      processed pseudogenes
        "translated_processed_pseudogene":"pseudo",              # Pseudogenes      processed pseudogenes
        "translated_unprocessed_pseudogene":"pseudo",            # Pseudogenes      unprocessed pseudogenes
                "TR_C_gene":"imm",                                       # Immunoglobulin/T-cell receptor gene segments       protein coding segments
        "IG_J_pseudogene":"imm",                                 # Immunoglobulin/T-cell receptor gene segments     pseudogenes
        "IG_V_gene":"imm",                                       # Immunoglobulin/T-cell receptor gene segments     protein coding segments
        "IG_V_pseudogene":"imm",                                 # Immunoglobulin/T-cell receptor gene segments     pseudogenes
        "known_ncrna":"lnc",                                     # Long non-coding RNA genes 
        "lincRNA":"lnc",                                         # Long non-coding RNA genes 
        "miRNA":"srna",                                          # Small non-coding RNA genes 
        "misc_RNA":"srna",                                       # Small non-coding RNA genes 
        "Mt_rRNA":"srna",                                        # Small non-coding RNA genes 
        "Mt_tRNA":"srna",                                        # Small non-coding RNA genes 
        "non_coding":"lnc",                                      # Long non-coding RNA genes 
        "polymorphic_pseudogene":"pseudo",                       # Pseudogenes      polymorphic pseudogenes
        "processed_pseudogene":"pseudo",                         # Pseudogenes      processed pseudogenes
        "processed_transcript":"lnc",                            # Long non-coding RNA genes   
        "protein_coding":"coding",                               # Protein-coding genes
        "pseudogene":"pseudo",                                   # Pseudogenes      pseudogenes
        "rRNA":"srna",                                           # Small non-coding RNA genes
        "sense_intronic":"lnc",                                  # Long non-coding RNA genes
        "sense_overlapping":"lnc",                               # Long non-coding RNA genes
        "snoRNA":"srna",                                         # Small non-coding RNA genes
        "snRNA":"srna",                                          # Small non-coding RNA genes
        "TEC":"lnc",                                             # Long non-coding RNA genes
        "transcribed_processed_pseudogene":"pseudo",             # Pseudogenes      processed pseudogenes
        "transcribed_unitary_pseudogene":"pseudo",               # Pseudogenes      unitary pseudogenes
        "transcribed_unprocessed_pseudogene":"pseudo",           # Pseudogenes      processed pseudogenes
        "translated_processed_pseudogene":"pseudo",              # Pseudogenes      processed pseudogenes
        "translated_unprocessed_pseudogene":"pseudo",            # Pseudogenes      unprocessed pseudogenes
        "TR_C_gene":"imm",                                       # Immunoglobulin/T-cell receptor gene segments       protein coding segments
        "TR_D_gene":"imm",                                       # Immunoglobulin/T-cell receptor gene segments       protein coding segments
        "TR_J_gene":"imm",                                       # Immunoglobulin/T-cell receptor gene segments       protein coding segments
        "TR_J_pseudogene":"imm",                                 # Immunoglobulin/T-cell receptor gene segments       pseudogenes
        "TR_V_gene":"imm",                                       # Immunoglobulin/T-cell receptor gene segments       protein coding segments
        "TR_V_pseudogene":"imm",                                 # Immunoglobulin/T-cell receptor gene segments       pseudogenes
        "unitary_pseudogene":"imm",                              # Immunoglobulin/T-cell receptor gene segments       pseudogenes
        "unprocessed_pseudogene":"pseudo",                       # Pseudogenes      unprocessed pseudogenes
		"scaRNA":"srna",
		"sRNA":"srna",
		"ribozyme":"lnc",
		"macro_lncRNA":"lnc",
		"vaultRNA":"srna",
		"bidirectional_promoter_lncrna":"lnc",
		"-":"unknown",
		"ncRNA":"lnc",
		"transposable_element":"transposable"
		}


##=== IUPAC
hiupac= {
		'A'    :'A',
		'C'    :'C',
		'G'    :'G',
		'T'    :'T',
		'U'    :'U',
		'AG'   :'R',
		'CT'   :'Y',
		'CU'   :'Y',
		'GT'   :'K',
		'GU'   :'K',
		'AC'   :'M',
		'CG'   :'S',
		'AT'   :'W',
		'AU'   :'W',
		'CGT'  :'B',
		'CGU'  :'B',
		'AGT'  :'D',
		'AGU'  :'D',
		'ACT'  :'H',
		'ACU'  :'H',
		'ACG'  :'V',
		'ACGU' :'N',
		'ACGT' :'N',
		'-'    :'-',
		#'AN':'N',
		#'CN':'N',
		#'GN':'N',
		#'NT':'N',
		#'NU':'N',
		}

