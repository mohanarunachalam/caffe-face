import numpy as np
from numpy import linalg as LA
import scipy.io as sio
from joblib import Parallel, delayed

mat_contents = sio.loadmat('LFW_Feature.mat')
face_names = mat_contents['list']
face_names = [str(obj[0][0]) for obj in face_names]
features = np.transpose(mat_contents['feature'])
features = features.tolist()

featuresMap = dict(zip(face_names, features))

def genPairsDistances(fileName):
	f = open(fileName)
	content = f.readlines()
	f.close()

	numLines = int(content[0])
	content = [x.rstrip('\n').split('\t') for x in content[1:]]
	# content = content[:100]
	peopleDict = dict([tuple((x[0],int(x[1]))) for x in content])

	peopleList = [k+'_%0.4d'%(i+1)+'.jpg' for k,v in sorted(peopleDict.iteritems()) for i in range(v)]
	countArray = np.cumsum(np.array([0]+[v for k,v in sorted(peopleDict.iteritems())]))
	embeddings = np.array([featuresMap[person]/LA.norm(featuresMap[person]) for person in peopleList])

	sameDistances = []
	diffDistances = []
	numPeople = np.size(countArray)-1
	for personIdx in range(numPeople):
		personStartIdx = countArray[personIdx]
		personEndIdx = countArray[personIdx+1]
		diffEmbeddings = embeddings[personEndIdx:]
		for picIdx in range(personStartIdx,personEndIdx-1):
			thisEmbedding = embeddings[picIdx]
			sameEmbeddings = embeddings[picIdx+1:personEndIdx]
			sameDistances = sameDistances + LA.norm((thisEmbedding - sameEmbeddings), axis=1).tolist()
			diffDistances = diffDistances + LA.norm((thisEmbedding - diffEmbeddings), axis=1).tolist()

		thisEmbedding = embeddings[personEndIdx-1]
		diffDistances = diffDistances + LA.norm((thisEmbedding - diffEmbeddings), axis=1).tolist()

	return np.array(sameDistances), np.array(diffDistances)

def getDistances(fileName):
	f = open(fileName)
	content = f.readlines()
	f.close()

	numSamePairs =  int(content[0])
	numDiffPairs = len(content) - numSamePairs - 1
	content = [x.rstrip('\n').split('\t') for x in content]

	pairs1 = [x[0]+'_%0.4d'%int(x[1])+'.jpg' for x in content[1:1+numSamePairs]]
	pairs2 = [x[0]+'_%0.4d'%int(x[2])+'.jpg' for x in content[1:1+numSamePairs]]

	pairs1 = pairs1 + [x[0]+'_%0.4d'%int(x[1])+'.jpg' for x in content[1+numSamePairs:]]
	pairs2 = pairs2 + [x[2]+'_%0.4d'%int(x[3])+'.jpg' for x in content[1+numSamePairs:]]

	features1 = np.array([featuresMap[person]/LA.norm(featuresMap[person]) for person in pairs1])
	features2 = np.array([featuresMap[person]/LA.norm(featuresMap[person]) for person in pairs2])

	diff = LA.norm(features1-features2,axis=1)
	return diff[:numSamePairs], diff[numSamePairs:]

def get_errors(sameDistances, diffDistances, thresholds):
	idx = 0
	errors = np.empty(len(thresholds))
	for thres in thresholds:
		sameError = (sameDistances > thres)
		diffError = (diffDistances <= thres)
		totalError = np.sum(sameError) + np.sum(diffError)
		errors[idx] = totalError
		idx = idx + 1
	errors_ratio = errors/float(np.size(sameDistances)+np.size(diffDistances))
	return errors,errors_ratio

def train_test_pairs(trainFile,testFile,genPairs=False):
	if(genPairs):
		print("In True")
		sameDistancesTrain, diffDistancesTrain = genPairsDistances(trainFile)
		sameDistancesTest, diffDistancesTest = genPairsDistances(testFile)
	else:
		print("In False")
		sameDistancesTrain, diffDistancesTrain = getDistances(trainFile)
		sameDistancesTest, diffDistancesTest = getDistances(testFile)
	
	thresholds = np.arange(0,1,0.001)
	errors, errors_ratio = get_errors(sameDistancesTrain, diffDistancesTrain, thresholds)
	min_error_idx = np.argmin(errors)
	best_thres = thresholds[min_error_idx]
	print(best_thres)
	min_errors_train = errors[min_error_idx]
	min_errors_ratio_train = errors_ratio[min_error_idx]

	errors, errors_ratio = get_errors(sameDistancesTest, diffDistancesTest, [best_thres])
	min_errors_test = errors[0]
	min_errors_ratio_test = errors_ratio[0]

	return min_errors_train, min_errors_ratio_train, min_errors_test, min_errors_ratio_test

errors_train, errors_ratio_train, errors_test, errors_ratio_test = train_test_pairs('pairsDevTrain.txt','pairsDevTest.txt',genPairs=False)
print(errors_train)
print(errors_ratio_train)
print(errors_test)
print(errors_ratio_test)

errors_train, errors_ratio_train, errors_test, errors_ratio_test = train_test_pairs('peopleDevTrain.txt','peopleDevTest.txt',genPairs=True)
print(errors_train)
print(errors_ratio_train)
print(errors_test)
print(errors_ratio_test)