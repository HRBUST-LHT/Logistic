import numpy as np


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def classifyVector(inX, weights):
    # use featureVector & weights to calculate the value of sigmoidFunction, and classify the result
    prob = sigmoid(sum(inX*weights))
    # print("\nprob: \n" + str(prob))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def gradAscent(dataMatrix, classLabels, numIter = 500):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    alpha = 0.001

    for j in range(numIter):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]

    return weights


def improvedStocGradAscent(dataMatrix, classLabels, numIter = 150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + i + j) + 0.001
            randomIndex = int(np.random.uniform(0, len(dataIndex)))
            chooseOne = dataIndex[randomIndex]
            h = sigmoid(sum(dataMatrix[chooseOne]*weights))
            error = classLabels[chooseOne] - h
            weights = weights + alpha*error * dataMatrix[chooseOne]
            del(chooseOne)

    return weights


def horseTest():
    frTrain = open('/Users/lihaotian/PycharmProjects/Logistic/horseColicTraining.txt', 'r')
    frTest = open('/Users/lihaotian/PycharmProjects/Logistic/horseColicTest.txt', 'r')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        # print("\ncurrLine: \n" + str(currLine))
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainingWeights1 = gradAscent(np.array(trainingSet), trainingLabels)
    # print("\ntrainingWeights_gradAscent: \n" + str(trainingWeights1))
    trainingWeights2 = improvedStocGradAscent(np.array(trainingSet), trainingLabels, 500)
    # print("\ntrainingWeights_improvedStocGradAscent: \n" + str(trainingWeights2))

    errorCount1 = 0
    errorCount2 = 0
    numTestVec = 0

    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        # print("\ncurrLine: \n" + str(currLine))
        lineArr = []
        for i in range(21):
            lineArr. append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainingWeights1)) != int(currLine[21]):
            errorCount1 += 1
        if int(classifyVector(np.array(lineArr), trainingWeights2)) != int(currLine[21]):
            errorCount2 += 1

    errorRate1 = (float(errorCount1) / numTestVec)
    print '\nthe error rate1 is %f' %  errorRate1
    errorRate2 = (float(errorCount2) / numTestVec)
    print 'the error rate2 is %f\n' % errorRate2

    # return errorRate1
    return errorRate2

def multiTest():
    numTest = 10
    errorSum = 0.0
    for k in range(numTest):
        errorSum += horseTest()
    print '\nafter %d iterations the average error is %f' % (numTest, errorSum / float(numTest))



if __name__ == '__main__':
    multiTest()
    # horseTest()