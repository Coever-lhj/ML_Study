
# coding: utf-8

# In[8]:


# Numpy : 과학용 계산 패키지
from numpy import *
# operator : kNN 알고리즘에서 정렬 처리를 하기 위한 패키지
import operator

# 데이터의 집합과 분류 항목을 생성하는 함수
def createDataSet():
    group = array([[1.0,1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# In[91]:


# k-최근접 이웃 알고리즘
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems, key = operator.itemgetter(1), reverse = True)
    
    return sortedClassCount[0][0]


# In[93]:


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines()) # 파일의 줄 수 구하기
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

