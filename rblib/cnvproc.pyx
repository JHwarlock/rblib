import sys
import numpy as np


class CNVproc(object):
    def __init__(self,segments,totseglen=2875001522):# autosomal = 2875001522, autosomal + X = 3031042417, autosomal + X + Y = 3088269832 
        """
        [chr1,start,end,segratio,meansegratio,ploidy], chr1,1,1000, 1.1, 1.1, 2.0
        """
        self.totseglen = totseglen
        self.segments  = segments
        self.numsegments = len(segments)
    def wploidy(self,idx=5,ploidy=2.0,tgain=2.5,tloss=1.5): # weighted_ploidy, Mean chromosome copy number
        """
        default,idx=5  use ploidy estimated
        ref-ploidy = 2
        """
        retploidy = 0.0
        retainlen = self.totseglen
        for i in range(self.numsegments):
            tmpploidy = self.segments[i][idx]
            if tloss < tmpploidy < tgain: continue
            if tmpploidy > tgain: tmpploidy = 3.0
            else:
                tmpploidy = 1.0
            segmentlen = self.segments[i][2] - self.segments[i][1] + 1.0
            #print segmentlen
            #print segmentlen/self.totseglen
            retploidy += tmpploidy * (segmentlen/self.totseglen)
            retainlen -= segmentlen
        retploidy += ploidy * (retainlen/self.totseglen)
        return retploidy

    def wgii(self,tgain=2.5,tloss=1.5,chrlen=None,numchrs=22.0):
        ## gii score weighted by chromosome length
        ## Replication stress links structural and numerical cancer chromosomal instability, Nature volume 494, pages 492–496 (28 February 2013)
        
        ## update 2021-07-17  add the wgii deletion only, wgii amplification only, and deletion per million and amplification per million

        segmentsgain = [] # weighted region
        segmentsloss = []

        rawgain      = [] # raw region
        rawloss      = []
        for chrom,start,end,segratio,meansegratio,ploidy in self.segments:
            tmpchrlen = chrlen[chrom]
            if meansegratio > tgain:
                segmentsgain.append((end-start+1.0)/tmpchrlen)
                rawgain.append(end-start+1.0)
            elif meansegratio < tloss:
                segmentsloss.append((end-start+1.0)/tmpchrlen)
                rawloss.append(end-start+1.0)

        segmentsgain = np.asarray(segmentsgain)
        segmentsloss = np.asarray(segmentsloss)
        wgii = np.sum(segmentsgain) + np.sum(segmentsloss)
        wgiidel = np.sum(segmentsloss)
        wgiiamp = np.sum(segmentsgain)
        # add return wgii, wgiidel, wgiiamp, delregion, ampregion 
        return wgii / numchrs,wgiidel / numchrs, wgiiamp / numchrs, np.sum(rawloss)/ 1000000.0,  np.sum(rawgain)/ 1000000.0


if __name__ == "__main__":
    print("[INFO] TEST: ...")
    segments = [["chr1",1,1000,-1.0,-1.0,1.0],["chr2",1,3000,-1.0,-1.0,3.0]]
    cnvproc = CNVproc(segments,totseglen=8000)
    print(cnvproc.wploidy())
    print(cnvproc.wgii(chrlen={"chr1":2000,"chr2":6000},numchrs=2.0))
    
