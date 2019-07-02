import sys
import simplejson as json


def iterdict(h,iterm_b="",hn={}):
	for iterm in h:
		print iterm_b
		print iterm
		termout = iterm_b + "=>" + iterm
		if type(h[iterm]) == dict:
			iterdict(h[iterm],iterm_b=termout,hn=hn)
		else:
			value = h[iterm]
			hn[termout+"=>"+iterm] = value
	
	return hn

"""
a = {
		"11":3,
		"12":4,
		"19":8,
		"L1":{
			"A":7,
			"B":8,
			"C":9,
			"L2":{
				"AA":19,
				"CA":18,
				"FC":21,
				},
			},
		"1111":78,
		"L3":{"AA":12,
			"BB":17,
			},
		}

print iterdict(a)
"""
if __name__ == "__main__":
	infile = file(sys.argv[1],"r")

	obj = json.load(infile)
	hn = iterdict(obj)
	for key in hn:
		if "days_to_death" in key:
			print key+ "\t" +hn[key]

