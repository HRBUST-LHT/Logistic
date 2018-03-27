import numpy as np

# sigmoid function
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def classifyVector(inX, weights):
    # use featureVector & weights to calculate the value of sigmoidFunction, and classify the result
    prob = sigmoid(sum(inX*weights))
    print("\nprob: \n" + str(prob))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def improvedStocGradAscent(dataMatrix, classLabels, numIter = 150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    print("\nn: \n" + str(n))
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + i + j) + 0.01
            randomIndex = int(np.random.uniform(0, len(dataIndex)))
            chooseOne = dataIndex[randomIndex]
            h = sigmoid(sum(dataMatrix[chooseOne]*weights))
            error = classLabels[chooseOne] - h
            weights = weights + alpha*error*dataMatrix[chooseOne]
            del(chooseOne)

    return weights


def horseTest():
    frTrain = open('/Users/lihaotian/PycharmProjects/Logistic/horseColicTraining.txt', 'r')
    frTest = open('/Users/lihaotian/PycharmProjects/Logistic/horseColicTest.txt', 'r')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainingWeights = improvedStocGradAscent(np.array(trainingSet), trainingLabels, 500)
    print("\ntrainingWeights: \n" + str(trainingWeights))
    errorCount = 0
    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        print("\ncurrLine: \n" + str(currLine))
        lineArr = []
        for i in range(21):
            lineArr. append(float(currLine[i]))
        # if int(classifyVector(np.array(lineArr), trainingWeights)) != int(currLine[21]):
        #     errorCount += 1
            if sigmoid(np.array(lineArr[i])) != int(currLine[21]):
                errorCount += 1

    errorRate = (float(errorCount) / numTestVec)
    print 'the error rate is %f' %  errorRate
    return errorRate


def multiTest():
    numTest = 10
    errorSum = 0.0
    for k in range(numTest):
        errorSum += horseTest()
    print 'after %d iterations the average error is %f' % (numTest, errorSum / float(numTest))



if __name__ == '__main__':
    # multiTest()
    horseTest()