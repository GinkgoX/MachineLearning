import numpy as np
import operator

'''
Function : 	createDataSet()
	Args: 	None
	Rets: 	group, dataset 
			lables, classes
'''
def createDataSet():
	group = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
	lables = ['Ramatic', 'Ramatic', 'Action', 'Action', 'Action']
	return group, lables

'''
Function : 	kNN(test, group, lables, k)
	Args :	test, test set
			group, train set
			lables, classes
			k, k nearest setting
	Rets :	pred[0][0], predict class
'''
def kNN(test, group, lables, k):
	dataSize = group.shape[0]
	#tile: to copy data as Array_m*n
	diff = np.tile(test, (dataSize, 1)) - group
	sqrdiff = diff**2
	#sum(axis = 1) : sum as row
	sumdiff = sqrdiff.sum(axis = 1)
	dist = sumdiff**0.5
	#argsort : return the index of dist order, [3, 1, 2] -> [1, 2, 0] asceeding order
	dist_order = dist.argsort()
	classes = {}
	for i in range(k):
		voteLable = lables[dist_order[i]]
		classes[voteLable] = classes.get(voteLable, 0) + 1
	#sorted : for any iterator, return list
	pred = sorted(classes.items(), key = operator.itemgetter(1), reverse = True)
	return pred[0][0] 

if __name__ == '__main__':
	group, lables = createDataSet()
	test = [18, 90]
	print('test : ', test)
	pred_class = kNN(test, group, lables, 3)
	print('predict class : ', pred_class)
