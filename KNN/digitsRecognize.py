import operator
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

'''
Function :	img2vector(filename)
	Description :	to covert img(in filename) to vector
	Args :	filename
	Rets :	vectorImg
'''

def img2vector(filename):
	vectorImg = np.zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		line = fr.readline()
		for j in range(32):
			vectorImg[0, 32*i + j] = int(line[j])
	return vectorImg

'''
Function : train()
	Description :	use kNN to train and test digits
	Args :	None
	Rets :	None
'''
def train():
	labels = []
	trainSet = listdir('./digits/trainSet')
	numTrain = len(trainSet)
	trainMatrix = np.zeros((numTrain, 1024)) #32*32 img size
	for i in range(numTrain):
		filename = trainSet[i]
		label = int(filename.split('_')[0])
		labels.append(label)
		trainMatrix[i, :] = img2vector('./digits/trainSet/%s'%(filename))
	neigh = kNN(n_neighbors = 3, algorithm = 'auto')
	neigh.fit(trainMatrix, labels)
	testSet = listdir('./digits/testSet')
	errorCount = 0.0
	numTest = len(testSet)
	for i in range(numTest):
		filename = testSet[i]
		label = int(filename.split('_')[0])
		vectorImg = img2vector('./digits/testSet/%s'%(filename))
		predLabel = neigh.predict(vectorImg)
		print('label: %d  vs  predLabel: %d'%(label, predLabel))
		if(label != predLabel):
			errorCount += 1.0
	print('Error Rate : %f%%'%(errorCount / numTest * 100))

if __name__ == '__main__':
	train()


