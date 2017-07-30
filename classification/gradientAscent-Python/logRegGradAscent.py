# Classification task - logistic regression
# batchGradAscent, stocGradAscent, minibatchGradAscent, batchGradAscentMultiProcess
# Python multiprocess paralleling of batchGradAscent

# @HYPJUDY 2017.6.21
# Details: https://hypjudy.github.io/2017/06/23/regression-classification-kaggle/
# Code takes inspiration from @hjc


# encoding=utf-8
# -*- coding:gb2312 -*-
from math import exp
import random
import os
import time
from multiprocessing import Process, Array, Lock


######## --------------------- Define Parameters ------------------ ########
TRAINPATH = '/home/data/xgboost/ML_HW2/data/train_standard10_9.txt'
VALPATH = '/home/data/xgboost/ML_HW2/data/val_standard10_1.txt'
TESTPATH = '/home/data/xgboost/ML_HW2/data/test_data_standard.txt'

N_ITER = 1
ALPHA = 0.1
N_FEATURE = 132 # features of train data range from 1 to 201
                # but features of test data range from 1 to 132


######## --------------------------- Utils ------------------------ ########

def loadDataSet(path):
	dataMat = [] # string of data (features)
	labelMat = []
	with open(path, 'r') as fr:
		for line in fr.readlines():
			index = line.find(' ')
			dataMat.append(line[index+1:])
			labelMat.append(int(line[0]))
	return dataMat, labelMat

def parseData(dataStr):
# parse string of data (features) into a dict: {index:val,...}
	dataDict = {}
	lineArr = dataStr.strip().split()
	for item in lineArr:
		index, val = item.split(':')
		if int(index) > N_FEATURE:
			continue
		dataDict[int(index)] = float(val)
	return dataDict

def predict(weights, dataDict):
	z = 0.0
	for index in dataDict:
		z = z + weights[index] * dataDict[index]
	return z

def sigmoid(x):
	# return expit(x)
	# avoid 'RuntimeWarning: overflow encountered in exp'
	if x > 30:
		return 1.0
	if x < -30:
		return 0.0
	return 1.0/(1 + exp(-x))

def saveWeights(path, weights):
	with open(path, 'w') as fw:
		for i in range(len(weights)):
			fw.write(str(weights[i]) + "\n")

def loadWeights(path):
	weights = [random.random() for j in range(N_FEATURE + 1)]
	i = 0
	with open(path, 'r') as fr:
		for weight in fr.readlines():
			weights[i] = float(weight)
			i = i + 1
	return weights

def calcAcc(weights, dataMat, labelMat, iter):
	hitNum = 0
	sampleNum = len(dataMat)
	for i in range(sampleNum):
		dataDict = parseData(dataMat[i])
		z = predict(weights, dataDict)
		if ((z >= 0 and labelMat[i] == 1) or \
			(z < 0 and labelMat[i] == 0)):
			hitNum = hitNum + 1
	acc = float(hitNum) / sampleNum
	print 'iter', iter, ': accuracy =', \
		hitNum, '/' , sampleNum, '=', acc

def testDataSet(weights, dataMat, path):
	num = 0
	with open(path, 'w') as fw:
		fw.write('id,label\n')
		for i in range(len(dataMat)):
			data = parseData(dataMat[i])
			y = sigmoid(predict(weights, data))
			if(y>=0.5):
				y=1
			else:
				y=0
			fw.write(str(num) + ',' + str(y) + '\n')
			num = num + 1



######## ------------- Different Gradient Ascent Method ----------- ########

def batchGradAscent(dataMat, labelMat, valDataMat, valLabelMat, numIter = N_ITER):
	alpha = ALPHA
	weights = [random.random() for i in range(N_FEATURE + 1)] # random init
	totalTime = 0.0
	for iter in range(numIter):
		start = time.time()
		_weights = [weights[i] for i in range(N_FEATURE + 1)] # copy
		for i in range(len(dataMat)):
			dataDict = parseData(dataMat[i])
			error = labelMat[i] - sigmoid(predict(weights, dataDict))
			for index in dataDict:
				_weights[index] = _weights[index] + alpha*error*dataDict[index]
		weights = [_weights[i] for i in range(N_FEATURE + 1)] # update
		end = time.time()
		totalTime = totalTime + float(end - start)
		calcAcc(weights, valDataMat, valLabelMat, iter)
	print 'batchGradAscent iter cost:', float(totalTime) / numIter, 's in average.'
	return weights

def stocGradAscent(dataMat, labelMat, valDataMat, valLabelMat, numIter = N_ITER):
	alpha = ALPHA
	weights = [random.random() for i in range(N_FEATURE + 1)]
	totalTime = 0.0
	for iter in range(numIter):
		start = time.time()
		for i in range(len(dataMat)):
			randIndex = int(random.uniform(0,len(dataMat)))
			dataDict = parseData(dataMat[randIndex])
			error = labelMat[randIndex] - sigmoid(predict(weights, dataDict))
			for index in dataDict:
				weights[index] = weights[index] + alpha*error*dataDict[index]
		end = time.time()
		totalTime = totalTime + float(end - start)
		calcAcc(weights, valDataMat, valLabelMat, iter)
	print 'stocGradAscent iter cost:', float(totalTime) / numIter, 's in average.'
	return weights

def minibatchGradAscent(dataMat, labelMat, valDataMat, valLabelMat,\
 batch = 10, numIter = N_ITER):
	alpha = ALPHA
	weights = [random.random() for i in range(N_FEATURE + 1)]
	totalTime = 0.0
	for iter in range(numIter):
		start = time.time()
		for j in range(len(dataMat) / batch):
			randIndex = int(random.uniform(0,len(dataMat)-batch))
			_weights = [weights[i] for i in range(N_FEATURE + 1)]
			for i in range(batch):
				dataDict = parseData(dataMat[randIndex+i])
				error = labelMat[randIndex+i] - sigmoid(predict(weights, dataDict))
				for index in dataDict:
					_weights[index] = _weights[index] + alpha*error*dataDict[index]
			weights = [_weights[i] for i in range(N_FEATURE + 1)]
		end = time.time()
		totalTime = totalTime + float(end - start)
		calcAcc(weights, valDataMat, valLabelMat, iter)
	print 'minibatchGradAscent iter cost:', float(totalTime) / numIter, 's in average.'
	return weights


def calcGrad(_weights, alpha, dataMat, labelMat, lock):
	t_weights = [_weights[i] for i in range(N_FEATURE + 1)]
	sums = [0 for i in range(N_FEATURE + 1)]
	for i in range(len(dataMat)):
		dataDict = parseData(dataMat[i])
		error = labelMat[i] - sigmoid(predict(t_weights, dataDict))
		for index in dataDict:
			sums[index] = sums[index] + alpha*error*dataDict[index] # Ascent
	with lock:
		for i in range(N_FEATURE + 1):
			_weights[i] += sums[i]

def batchGradAscentMultiProcess(dataMat, labelMat, valDataMat, valLabelMat, \
	processNum = 18, numIter = N_ITER):
	_weights = Array('f', range(N_FEATURE + 1))
	alpha = ALPHA
	weights = [random.random() for i in range(N_FEATURE + 1)] # random init
	
	sampleNum = len(dataMat)
	if sampleNum < processNum:
		processNum = sampleNum
	step = sampleNum / processNum # workload of each process
	
	totalTime = 0.0
	for iter in range(numIter):
		start = time.time()
		lock = Lock()
		for i in range(len(weights)): _weights[i] = weights[i]

		processes = [] # list
		for i in range(0, sampleNum, step):
			if i + step > sampleNum:
				continue
			process = Process(target=calcGrad, \
				args=(_weights, alpha, dataMat[i:i+step], labelMat[i:i+step], lock))
			processes.append(process)

		for i in range(len(processes)):
			# print 'Process ', i, ' started.'
			processes[i].start()

		# join(): Block the calling thread until the process
		# whose join() method is called terminates
		for i in range(len(processes)):
			processes[i].join()
			# print 'Process ', i, 'ended.'

		weights = [_weights[i] for i in range(N_FEATURE + 1)]
		end = time.time()
		totalTime = totalTime + float(end - start)
		calcAcc(weights, valDataMat, valLabelMat, iter)
	print 'batchGradAscentMultiProcess iter cost:', float(totalTime) / numIter,\
	's in average. processNum:', processNum
	return weights



######## ------------------------ Main Process --------------------- ########
# run by command: nohup python -u logRegGradAscent.py &

if __name__ ==  '__main__':
	start = time.time()
	# step 1: load data
	dataMat, labelMat = loadDataSet(TRAINPATH)
	valDataMat, valLabelMat = loadDataSet(VALPATH)
	# testDataMat, ids = loadDataSet(TESTPATH)
	
	# step 2: train
	## Group 1: Compare Different Method:
	# weights = gradAscent(dataMat, labelMat, valDataMat, valLabelMat)
	# weights = stocGradAscent(dataMat, labelMat, valDataMat, valLabelMat)
	# weights = minibatchGradAscent(dataMat, labelMat, valDataMat, valLabelMat)
	# weights = batchGradAscentMultiProcess(dataMat, labelMat, valDataMat, valLabelMat)
	
	## Group 2: Compare Different processNum in parallel:
	# weights = batchGradAscentMultiProcess(dataMat, labelMat, valDataMat, valLabelMat, 1)
	# weights = batchGradAscentMultiProcess(dataMat, labelMat, valDataMat, valLabelMat, 2)
	# weights = batchGradAscentMultiProcess(dataMat, labelMat, valDataMat, valLabelMat, 4)
	# weights = batchGradAscentMultiProcess(dataMat, labelMat, valDataMat, valLabelMat, 8)
	# weights = batchGradAscentMultiProcess(dataMat, labelMat, valDataMat, valLabelMat, 16)
	weights = batchGradAscentMultiProcess(dataMat, labelMat, valDataMat, valLabelMat, 32)
	# weights = batchGradAscentMultiProcess(dataMat, labelMat, valDataMat, valLabelMat, 64)
	# weights = batchGradAscentMultiProcess(dataMat, labelMat, valDataMat, valLabelMat, 128)

	# step 3: test
	# testDataSet(weights, testDataMat, 'result.txt')
	# saveWeights('weights.txt', weights)

	end = time.time()
	print 'Total Process Time:', float(end - start), 's.'
