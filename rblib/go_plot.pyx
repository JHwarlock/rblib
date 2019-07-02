import sys
import commands
import os
###getAmigoTree: Currently only the amigo (v2) and advanced (v1) modes are supported! (not quickgo or single)
def RamiGO_exec(goid_list,fig_prefix="amigo",colors = None):
	if colors == None:
		colors = "\"pink\""
	else:
		colors = "\""+"\",\"".join(colors)+"\""
	if goid_list:
		goid_list = "\""+"\",\"".join(goid_list)+"\""
	else:
		sys.stderr.write("[WARN] No GO iterms found\n")
		return 0
	Rcode = """
	library(RamiGO)
	goIDs <- c(%s)
	color <- c(%s)
	png()
	pngRes <- getAmigoTree(goIDs=goIDs, color=color, filename="%s", picType="png",modeType="advanced",webserver="http://amigo.berkeleybop.org/cgi-bin/amigo/visualize", saveResult=TRUE)
	dev.off()
	svg()
	svgRes <- getAmigoTree(goIDs=goIDs, color=color, filename="%s", picType="svg",modeType="advanced",webserver="http://amigo.berkeleybop.org/cgi-bin/amigo/visualize", saveResult=TRUE)
	dev.off()
	"""%(goid_list,colors,fig_prefix,fig_prefix)
	#print Rcode
	f = file(".tmp.amigo.R","w")
	f.write(Rcode)
	f.close()
	status,output = commands.getstatusoutput("which Rscript")
	print output
	status,output = commands.getstatusoutput(output+" .tmp.amigo.R")
	#os.remove(".tmp.amigo.R")
	sys.stderr.write("[R EXEC] status: %d, detail info: \n%s"%(status,output))
	if status:return 1
	return 0
if __name__ == "__main__":
	#pngRes <- getAmigoTree(goIDs=goIDs,color=color, filename="amigo", picType="png", modeType="advanced",webserver="http://amigo.berkeleybop.org/cgi-bin/amigo/visualize",saveResult=TRUE)
	goid_list = "GO:0030502","GO:0031072","GO:0031519","GO:0035518","GO:0042476","GO:0042826","GO:0044212","GO:0045892","GO:0045892","GO:0051572","GO:0060021","GO:0065001","GO:0070171","GO:0005096","GO:0005509","GO:0005515","GO:0005635","GO:0005737","GO:0005829","GO:0006915","GO:0042692","GO:0043066","GO:0043231","GO:0043547","GO:0048471","GO:0051149","GO:0000070","GO:0000278","GO:0000776","GO:0000777","GO:0005515","GO:0005634","GO:0005730","GO:0005829","GO:0007094","GO:0007126","GO:0008608","GO:0031145","GO:0051301","GO:0051436","GO:0051439","GO:0051983","GO:0071173","GO:0005634","GO:0005730","GO:0005737","GO:0005813","GO:0005819","GO:0005737","GO:0005813","GO:0051297","GO:0000288","GO:0000289","GO:0000932","GO:0001829","GO:0005515","GO:0005634","GO:0005737","GO:0005829","GO:0006351","GO:0006355","GO:0006417","GO:0010467","GO:0016070","GO:0016071","GO:0030014","GO:0031047","GO:2000036","GO:0000166","GO:0000288","GO:0000289","GO:0003723","GO:0003743","GO:0005515","GO:0005829","GO:0006412","GO:0006413","GO:0006446","GO:0008286","GO:0010467","GO:0016070","GO:0016071","GO:0016281","GO:0044267","GO:0000122","GO:0001932","GO:0001934","GO:0002327","GO:0002904","GO:0005085","GO:0005515","GO:0005737","GO:0009267","GO:0031929","GO:0032007","GO:0033138","GO:0043154","GO:2000973","GO:0004842","GO:0005634","GO:0006351","GO:0006355","GO:0008270","GO:0015630","GO:0021785","GO:0021952","GO:0030424","GO:0032880","GO:0042803","GO:0051493","GO:0000278","GO:0005215","GO:0005487","GO:0005515","GO:0005643","GO:0005654","GO:0005829","GO:0005975","GO:0006406","GO:0006606","GO:0006611","GO:0007077","GO:0008645","GO:0010467","GO:0010827","GO:0015758","GO:0016032","GO:0016070","GO:0016071","GO:0019221","GO:0044281","GO:0051726","GO:0055085","GO:0005097","GO:0005634","GO:0005769","GO:0005794","GO:0017137","GO:0032851","GO:0032880","GO:0000166","GO:0003690","GO:0003697","GO:0003723","GO:0005515","GO:0005634","GO:0006260","GO:0006396","GO:0000166","GO:0000184","GO:0000462","GO:0003735","GO:0005634","GO:0005730","GO:0005737","GO:0005829","GO:0006364","GO:0006412","GO:0006412","GO:0006413","GO:0006414","GO:0006415","GO:0006614","GO:0010467","GO:0015935","GO:0015935","GO:0016032","GO:0016070","GO:0016071","GO:0019058","GO:0019083","GO:0022627","GO:0031369","GO:0034101","GO:0042274","GO:0044267","GO:0000166","GO:0003723","GO:0005515","GO:0005737","GO:0006396","GO:0016607","GO:0043231","GO:0001525","GO:0001763","GO:0001843","GO:0005515","GO:0005634","GO:0005694","GO:0006298","GO:0006355","GO:0006368","GO:0010793","GO:0016491","GO:0018024","GO:0018024","GO:0030900","GO:0034728","GO:0035441","GO:0046914","GO:0048332","GO:0048701","GO:0048864","GO:0060039","GO:0060669","GO:0060977","GO:0097198","GO:0097198","GO:0000184","GO:0000184","GO:0000723","GO:0000781","GO:0004521","GO:0004521","GO:0004521","GO:0005515","GO:0005634","GO:0005697","GO:0005730","GO:0005737","GO:0005829","GO:0005829","GO:0006406","GO:0010467","GO:0016070","GO:0016071","GO:0035303","GO:0042162"
	print RamiGO_exec(goid_list)
	
