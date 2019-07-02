# -*- coding: utf-8 -*
# linear regress and  local weighted regress

# LOESS ~ LOWESS

#!/usr/bin/python
# please add your code here!
 
import re;
import sys;
import time;
import numpy as np
# x location; h:bandwith;xp,yp:data points(vectors)
def loess(x,h,xp,yp):
	# diff bands will change fit  
	w=np.exp(-0.5*((x-xp)/h)**2)/np.sqrt(2*np.pi*h**2);
	b=np.sum(w*xp)*np.sum(w*yp)-np.sum(w)*np.sum(w*xp*yp);
	b/=np.sum(w*xp)**2-np.sum(w)*np.sum(w*xp**2);
	a=(np.sum(w*yp)-b*np.sum(w*xp))/np.sum(w);
	return a+b*x;


def loess_test():
	x = np.linspace(0,3.14,100)
	y = np.sin(x)



