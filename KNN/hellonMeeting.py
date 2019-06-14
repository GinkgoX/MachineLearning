import numpy as np
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

'''
Function : file2matrix(filename)
	Description : 	to covert file into matrix
	Args:	filename
	Rets:	featureMatrix, the matrix format from file coverting
			labels, the label for info
'''

def file2matrix(filename):
	fread = open(filename)
	info = fread.readlines()
	featureMatrix = np.zeros((len(info), 3))
	labels = []
	index = 0
	for line in info:
		line = line.strip()
		listline = line.split('\t')
		featureMatrix[index, :] = listline[0:3]
		if listline[-1] == 'didntLike':
			labels.append(1)
		if listline[-1] == 'smallDoses':
			labels.append(2)
		if listline[-1] == 'largeDoses':
			labels.append(3)
		index += 1
	return featureMatrix, labels

'''
Function : normalize(featureMatrix)
	Description : to normalize data
	Args :	featureMatrix
	Rets : normFeatureMatrix
'''
def normalize(featureMatrix):
	#get every column minVal
	minVal = featureMatrix.min(0)
	#get every row maxVal
	maxVal = featureMatrix.max(0)
	ranges = maxVal - minVal
	normFeatureMatrix = np.zeros(np.shape(featureMatrix))
	row = normFeatureMatrix.shape[0]
	normFeatureMatrix = featureMatrix - np.tile(minVal, (row, 1))
	normFeatureMatrix = normFeatureMatrix / np.tile(ranges, (row, 1))
	return normFeatureMatrix

'''
Function : visualize(featureMatrix, labels)
	Description : to visualize data
	Args :	featureMatrix
			labels
	Rets :	None
'''

def visualize(featureMatrix, labels):
	font = FontProperties(size = 14)
	#fig: figure object, axs : subplot
	fig, axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = False, figsize = (10, 10))
	labelColors = []
	for i in range(len(labels)):
		if i == 1:
			labelColors.append('black')
		if i == 2:
			labelColors.append('orange')
		if i == 3:
			labelColors.append('red')
	#subplot(0,0) scatter, s : scatter size, alpha : transparency
	axs[0][0].scatter(x = featureMatrix[:,0], y = featureMatrix[:, 1], color = labelColors, s = 15, alpha = 0.5)
	axs_0_title = axs[0][0].set_title(u'Route vs Game', FontProperties = font)
	axs_0_x = axs[0][0].set_xlabel(u'Route (km/year)', FontProperties = font)
	axs_0_y = axs[0][0].set_ylabel(u'Game (hours/week)', FontProperties = font)
	plt.setp(axs_0_title, size = 9, weight = 'bold', color = 'red')
	plt.setp(axs_0_x, size = 7, weight = 'bold', color = 'black')
	plt.setp(axs_0_y, size = 7, weight = 'bold', color = 'black')
	
	#subplot(0,1) scatter, s : scatter size, alpha : transparency
	axs[0][1].scatter(x = featureMatrix[:,0], y = featureMatrix[:, 2], color = labelColors, s = 15, alpha = 0.5)
	axs_1_title = axs[0][1].set_title(u'Route vs Icecream', FontProperties = font)
	axs_1_x = axs[0][1].set_xlabel(u'Route (km/year)', FontProperties = font)
	axs_1_y = axs[0][1].set_ylabel(u'Icecream (g/week)', FontProperties = font)
	plt.setp(axs_1_title, size = 9, weight = 'bold', color = 'red')
	plt.setp(axs_1_x, size = 7, weight = 'bold', color = 'black')
	plt.setp(axs_1_y, size = 7, weight = 'bold', color = 'black')
	
	#subplot(1,0) scatter, s : scatter size, alpha : transparency
	axs[1][0].scatter(x = featureMatrix[:,1], y = featureMatrix[:, 2], color = labelColors, s = 15, alpha = 0.5)
	axs_2_title = axs[1][0].set_title(u'Game vs Icecream', FontProperties = font)
	axs_2_x = axs[1][0].set_xlabel(u'Game (hours/week)', FontProperties = font)
	axs_2_y = axs[1][0].set_ylabel(u'Icecream (g/week)', FontProperties = font)
	plt.setp(axs_2_title, size = 9, weight = 'bold', color = 'red')
	plt.setp(axs_2_x, size = 7, weight = 'bold', color = 'black')
	plt.setp(axs_2_y, size = 7, weight = 'bold', color = 'black')
	
	#set legend
	didntLike = mlines.Line2D([], [], color = 'black', marker = '.', markersize = 6, label = 'didntLike')
	smallDoses = mlines.Line2D([], [], color = 'orange', marker = '.', markersize = 6, label = 'smallDoses')
	largeDoses = mlines.Line2D([], [], color = 'red', marker = '.', markersize = 6, label = 'largeDoses')
	axs[0][0].legend(handles = [didntLike, smallDoses, largeDoses])
	axs[0][1].legend(handles = [didntLike, smallDoses, largeDoses])
	axs[1][0].legend(handles = [didntLike, smallDoses, largeDoses])
	plt.show()

'''
Function : kNN(test, featureMatrix, labels, k)
	Description : to use kNN algorithm predict test result
	Args:	test	#test vector
			featureMatrix
			labels
			k	k classes
	Rets :	pred_class
'''
def kNN(test, featureMatrix, labels, k):
	row = featureMatrix.shape[0]
	diff = np.tile(test, (row, 1)) - featureMatrix
	sqdiff = diff**2
	dist = sqdiff.sum(axis = 1)
	dist = dist**0.5
	dist_order = dist.argsort()
	classes = {}
	for i in range(k):
		voteLabel = labels[dist_order[i]]
		classes[voteLabel] = classes.get(voteLabel, 0) + 1
	pred = sorted(classes.items(), key = operator.itemgetter(1), reverse = True)
	return pred[0][0]

'''
Function : train()
	Description : to train test data and record result
	Args : None
	Rets : None
'''
def train():
	filename = 'info.txt'
	featureMatrix, labels = file2matrix(filename)
	normFeatureMatrix = normalize(featureMatrix)
	inRote = 0.1
	row = normFeatureMatrix.shape[0]
	numTest = int(inRote * row)
	errorcount = 0.0
	for i in range(numTest):
		result = kNN(normFeatureMatrix[i,:], normFeatureMatrix[numTest:row, :], labels[numTest:row], 4)
		print('pred : %d vs real : %d'%(result, labels[i]))
		if result != labels[i]:
			errorcount += 1.0
	print('Error rate : %f %%'%(errorcount / float(numTest)*100))

'''
Function : score()
	Description :	to score for input info
		Args :	None
		Rets :	None
'''
def score():
	filename = 'info.txt'
	featureMatrix, labels = file2matrix(filename)
	#get every column minVal
	minVal = featureMatrix.min(0)
	#get every row maxVal
	maxVal = featureMatrix.max(0)
	resultList = ['didntLike', 'smallDoses', 'largeDoses']
	normFeatureMatrix = normalize(featureMatrix)
	route = float(input('Enter your routing precent : '))
	game = float(input('Enter your gaming precent : '))
	iceCream = float(input('Enter your iceCreaming precent : '))
	test = np.array([route, game, iceCream])
	normTest = (test - minVal) / (maxVal - minVal)
	result = kNN(normTest, normFeatureMatrix, labels, 3)
	print('Score : %s'%(resultList[result - 1]))

if __name__ == '__main__':
	#train()
	score()
