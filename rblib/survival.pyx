import sys
import pandas as pd
import lifelines
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import NelsonAalenFitter
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from rblib import statplot,enrich
import numpy as np
def plot_survival(dframe,groupheader="",time="",censored="",right=1,fig_prefix="test_survival",xlabel="PFS(Days)",ylabel="Survival rate",dotest=0,filename="test",figsize=(5,4),doplot=1,maxtime=None):# use group diff
	if maxtime is None:
		maxtime = np.ceil(dframe[time].max())
	ixname = []
	ixname_median = []
	levels = list(set(dframe[groupheader]))
	for i in range(len(levels)):
		tmplevel = levels[i]
		ixname.append(tmplevel)
	groups = dframe[groupheader]
	
	kmf = KaplanMeierFitter()
	T   = dframe[time]
	E   = dframe[censored]
	if not doplot:
		for i in ixname:
			ix = (groups == i)
			kmf.fit(T[ix], E[ix],label=i)
			ixname_median.append(kmf.median_)

	if doplot:
		fig = plt.figure(dpi=300,figsize=figsize)
		ax = fig.add_subplot(111)

		for i in ixname:
			ix = (groups == i)
			kmf.fit(T[ix], E[ix],label=i)
			ixname_median.append(kmf.median_)
			kmf.plot(ax=ax,show_censors=False,ci_show=False,linewidth=2)
		#ax.grid(True)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_ylim(0,1)
		ax.set_xlim(0,maxtime)

		fig.tight_layout()
		fig_prefix = filename + "." + fig_prefix
		plt.savefig(fig_prefix+".png",format='png',dpi=300);plt.savefig(fig_prefix+".svg",format='svg',dpi=300);plt.clf();plt.close();
		sys.stderr.write("[INFO] png and svg output to '%s'\n"%fig_prefix)
	
		fig = plt.figure(dpi=300,figsize=figsize)
		ax = fig.add_subplot(111)
		naf = NelsonAalenFitter()
		for i in ixname:
			ix = (groups == i)
			naf.fit(T[ix],event_observed=E[ix],label=i)
			naf.plot(ax=ax,show_censors=False,ci_show=False,linewidth=2)
		ax.set_xlabel(xlabel)
		ax.set_ylabel("Risk")
		ax.set_xlim(0,maxtime)
		fig.tight_layout()
		plt.savefig(fig_prefix+".NA.png",format='png',dpi=300);plt.savefig(fig_prefix+".NA.svg",format='svg',dpi=300);plt.clf();plt.close();
		sys.stderr.write("[INFO] png and svg output to '%s'\n"%fig_prefix)
		band = 8
		fig = plt.figure(dpi=300,figsize=figsize)
		ax = fig.add_subplot(111)
		naf = NelsonAalenFitter()
		for i in ixname:
			ix = (groups == i)
			a = naf.fit(T[ix],event_observed=E[ix],label=i,alpha=0.05)
			naf.cumulative_hazard_.to_csv("%s.%s.csv"%(filename,i))
		#print naf.smoothed_hazard_(0.5)
		#print naf.confidence_interval_
			naf.plot_hazard(ax=ax,bandwidth=band,show_censors=False,ci_show=False,linewidth=2)
			ax.set_xlim(0,maxtime)
		fig.tight_layout()
		plt.savefig(fig_prefix+".NAH.png",format='png',dpi=300);plt.savefig(fig_prefix+".NAH.svg",format='svg',dpi=300);plt.clf();plt.close();
	# do pair wise logranktest
	ret = []
	#print(ixname)
	if len(ixname) >= 2:
		pass
	else:
		return ret
	
	for i in range(len(ixname)):
		#print(i)
		for j in range(i+1,len(ixname)):
			#print(j)
			xi = ixname[i]; xj=ixname[j]
			xi_median = ixname_median[i]; xj_median = ixname_median[j]
			assert xi != xj
			result = logrank_test(T[groups==xi],T[groups==xj],E[groups==xi],E[groups==xj],alpha=.95)
			snnumlevel1 = len(T[groups==xi])
			snnumlevel2 = len(T[groups==xj])
			ret.append([groupheader,xi,xj,snnumlevel1,snnumlevel2,xi_median,xj_median,result.test_statistic,result.p_value,1.0])
	return ret
from rblib.traits import TraitInfo
if __name__ == "__main__":
	ti = TraitInfo()
	filename = sys.argv[1]
	ti.parse(sys.argv[1])
	time_v = "Progression_free_survival"
	censored="Event_observed"
	rets = []

	for trait in ti.traitname:
		if ti.traittype[trait] != "category":continue
		if trait == "SN" or trait == "bcr_patient_uuid":continue
		if trait in [time_v,censored] and ti.traittype[trait] != str:
			continue
		else:
			ret = plot_survival(ti.dframe,groupheader=trait,time=time_v,censored=censored,xlabel="Months from procedure",ylabel="Disease free survival rate",right=1,dotest=1,fig_prefix=trait+".survival",filename=filename)
			if ret:
				rets.extend(ret)
	if rets:
		rets =  enrich.fdr_core(rets,9,8)
	f = open(sys.argv[1]+".logranktest.xls","w")
	f.write("## logrank test result\n")
	f.write("\t".join(["#TraitName","Level1","Level2","Level1_SampleNumbers","Level2_SampleNumbers","Level1_Median_survival_time","Level2_Median_survival_time","Statistic","p-value","q-value"])+"\n")
	for t in rets:
		f.write("\t".join(list(map(str,t)))+"\n")
	f.close()
	sys.stderr.write("[INFO] result output to '%s'\n"%("Total."+"logranktest.xls"))
	exit(0)
