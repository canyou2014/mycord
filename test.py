from numpy import *
import operator
# a = array([[1,3,4],[1,3,5],[2,3,4],[3,5,4]])
# test = array([2,2,4])
# asize = a.shape[0]
# dismat = tile(test,(asize,1))
# dismat -= a
# dissq = dismat**2
# sqdistance = dissq.sum(axis=1)
# distance = sqdistance**0.5
# c = distance.argsort()
# d = 0
# classCount={}
# for i in range(0,50):
#     a = 0 if i%3==0 else 1 if i%4 ==0 else 2
#     classCount[a] = classCount.get(a,0)+1;
# sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
# print classCount,"\n",sortedClassCount[0][0]
a = array([[1,0,0,1,1,0,1,0,1],[0.1,0.6,0.7,1,1,0,1,0,1]]).T
b = mat(a)
c = nonzero(b[:, 0])
print c[0]