from numpy import *
import csv
import random
def loadCsv(filename):
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def naivebayes():
    return
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated



def creatfeaturelist(dataset):
    featureSet = set([])
    for uni in dataset:
        featureSet = featureSet | set(uni)
        return list(featureSet)
def trainnb(trainMatrix,label,trainset):
    numTrainSet = len(trainMatrix)
    uniqueLabel = unique(label)
    numlabel = len(uniqueLabel)
    pabusive = {}
    pNum = {}
    pSum = {}
    pfea  = {}
    numWord = len(trainMatrix[0])
    for nl in uniqueLabel:
        for l in label:
            if l == nl:
                pabusive[nl] += 1
    for key in pabusive:
        pabusive[key] /= numlabel
    for i in range(numTrainSet):
        pNum[label[i]] += trainMatrix
        pSum[label[i]] += sum(trainMatrix)
    sum1 = 0
    ###################################
    for i in uniqueLabel:
        for j in range(numTrainSet):
            sum1 += trainset[j][i]
    for ul in uniqueLabel:
        pfea[ul] = log(pNum[ul]/pSum[ul])
    return pfea,pabusive
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]
def summarize(dataset):
    sumfea = 0.0
    summaries = [[count(attribute),mean(attribute), stdev(attribute)] for attribute in zip(*dataset)]
    for i in range(len(summaries)):
        sumfea += summaries[i][0]
    for i in range(len(summaries)):
        summaries[i][0] /= sumfea
    del summaries[-1]
    return summaries

def mean(numbers):
    return sum(numbers)/count(numbers) if count(numbers) != 0 else 0
def count(numbers):
    count = 0
    for i in numbers:
        if i != 0:
            count += 1
    return count
def stdev(numbers):
    avg = mean(numbers)

    variance = sum([pow(x-avg,2) * (x != 0) for x in numbers])/float(count(numbers)-1)
    return math.sqrt(variance) if (count(numbers) != 0 and count(numbers) != 1) else 0
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    numlabel = {}
    prolabel = {}
    sumlabel = 0.0
    for label, value in separated.items():
        numlabel[label] = len(value)
        sumlabel += len(value)
        summaries[label] = summarize(value)
    for label in separated:
        prolabel[label] = numlabel[label]/sumlabel
    return summaries,prolabel
def calculateProbability(x, mean, stdev):
    if (stdev-0) < 1e-20 :
        return 0.0
    else:
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, prolabel, testSet):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 0.0
        ddd = 0
        for i in range(len(classSummaries)):
            prob, mean, stdev = classSummaries[i]
            aaa = testSet[i]

            ddd = calculateProbability(aaa, mean, stdev)
            probabilities[classValue] += prob*ddd
            x = 133
        probabilities[classValue] *= prolabel[classValue]
    return probabilities
def predict(summaries, prolabel, testSet):
    probabilities = calculateClassProbabilities(summaries, prolabel, testSet)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestLabel = classValue
            bestProb = probability
    return bestLabel
def getPredictions(summaries,prolabel, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, prolabel, testSet[i])
        predictions.append(result)
    return predictions
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
def test(filename, percent):
    import time
    oldtime1 = time.time()
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, percent)
    # print (len(trainingSet),len(testSet))
    summaries, prolabel= summarizeByClass(trainingSet)
    oldtime2 = time.time()
    predictions = getPredictions(summaries, prolabel, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print accuracy
    endtime = time.time()
    return endtime-oldtime2, oldtime2-oldtime1
if __name__ == '__main__':
    import profile,time
    i = 0
    t1, t2 = 0, 0
    while i<100:
        t3,t4 = test('heh.csv', 0.67)
        t1 += t3
        t2 += t4
        i += 1
    print t1, t2

