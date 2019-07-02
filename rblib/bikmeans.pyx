import numpy as np

def disEclud(vecA,vecB):
	return np.sqrt(np.sum(np.power(vecA-vecB,2)))

def randCent(dataSet,k):
	n = np.shape(dataSet)[1]
	centroids = np.asmatrix(np.zeros((k,n)))
	for j in xrange(n):
		minJ = np.min(dataSet[:,j])
		rangeJ = np.float(np.max(dataSet[:,j])-minJ)
		centroids[:,j] = minJ+rangeJ*np.random.rand(k,1)
	return centroids


def kMeans(dataSet,k,distMeas=disEclud,createCent = randCent):
	m = np.shape(dataSet)[0]
	clusterAssment = np.asmatrix(np.zeros((m,2)))
	centroids = createCent(dataSet,k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in xrange(m):
			minDist = np.inf
			minIndex = -1
			for j in xrange(k):
				distJI = distMeas(centroids[j,:],dataSet[i,:])
				if distJI < minDist:
					minDist = distJI
					minIndex = j
			if clusterAssment[i,0] != minIndex:
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist**2
		#print centroids
		for cent in xrange(k):
			ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
			centroids[cent,:] = np.mean(ptsInClust,axis =0)
	return centroids,clusterAssment

def bikmeans(dataSet,k,distMeas=disEclud):
	m = np.shape(dataSet)[0]
	clusterAssment = np.asmatrix(np.zeros((m,2)))
	centroid0 = np.mean(dataSet,axis=0).tolist()
	centList = [centroid0]
	for j in range(m):
		clusterAssment[j,1] = distMeas(np.mat(centroid0),dataSet[j,:]) ** 2
	while (len(centList) < k+1):
		lowestSSE = np.inf
		for i in xrange(len(centList)):
			ptsInCurrCulster = dataSet[np.nonzero(clusterAssment[:,0].A == i)[0],:]
			centroidMat,splitClustAss = kMeans(ptsInCurrCulster,2,distMeas)
			sseSplit = np.sum(splitClustAss[:,1])
			sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A != i)[0],1])
			print "sseSplit, and sseNotSplit: ",sseSplit,sseNotSplit
			if (sseSplit+sseNotSplit) <= lowestSSE:
				bestCentToSplit = i
				bestNewCents  =  centroidMat
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
		bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
		#print 'bestCentToSplit is: ',bestCentToSplit
		#print 'len of bestClustAss: ',len(bestClustAss)
		#centList[bestCentToSplit] = bestNewCents[0:,1]
		#print "#####",str(bestNewCents[1,:])
		centList.append(bestNewCents[1,:])
		clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
	#print centList
	return centList,clusterAssment

if __name__ == "__main__":
	pt1 = np.random.normal(1, 0.2, (100,2))
	pt2 = np.random.normal(50, 0.2, (100,2))
	pt3 = np.random.normal(100, 0.2, (100,2))
	xy = np.concatenate((pt1, pt2, pt3))
	from scipy.cluster.vq import whiten
	xy = whiten(xy)
	cents,clusterass =  bikmeans(xy,3)	
	print cents


