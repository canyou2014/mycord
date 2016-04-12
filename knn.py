from numpy import *
import operator
import random

def knn(testSet, dataSet, labels):
    k = 6
    dataSetSize = dataSet.shape[0]
    sortedClassCount = []
    for i in range(len(testSet)):
        inX = testSet[i, :]
        diffMat = tile(inX, (dataSetSize,1)) - dataSet#qiu oushijuli
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        sortedDistIndicies = distances.argsort()
        classCount={}
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + sortedDistIndicies[i]
        sortedClassCount.append(sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0])

    return sortedClassCount

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0

    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def splitDataset(datingDataMat,datingLabels,percent):
    testData = []
    testlable = []
    label = list(datingLabels)
    datingData = ndarray.tolist(datingDataMat)
    testsize = int(datingDataMat.shape[0]*percent)
    for i in xrange(testsize):
        index = random.randrange(len(datingData))
        testData.append(datingData.pop(index))
        testlable.append(label.pop(index))
    return array(datingData),label,array(testData),testlable

def getPrediction(datingData,label,testData,testlabel,func):
    count = 0
    xlabel = []
    dis = func(testData,datingData,label)
    for i in xrange(len(testData)):
        if(dis[i] == testlabel[i]):
            count +=1
        xlabel.append(dis)
    return xlabel,count/float(len(testData))

def test(filename, percent):
    import getdata, time
    starttime = time.time()
    filename = 'heh.csv'
    dataset = array(getdata.loadCsv(filename))
    # datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    datingData,label,testData,testlabel = splitDataset(dataset[:,:-1],dataset[:,-1], 0.67)
    xlabel,accracy = getPrediction(datingData,label,testData,testlabel,knn)
    print accracy*100,'%'
    endtime = time.time()
    return endtime-starttime
if __name__ == '__main__':
    i = 0
    t1 = 0
    while i<100:
        t1 += test('heh1.csv', 0.67)
        i += 1
    print t1