import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []   #trainingTest
    labelMat = []  #each label of every data(total 100)
    #the path of testSet.txt
    fr = open('/Users/lihaotian/PycharmProjects/Logistic/testSet.txt', 'r')
    #read the content of txt file by line
    for line in fr.readlines():
        lineArray = line.strip().split()
        dataMat.append([1.0, float(lineArray[0]), float(lineArray[1])])
        labelMat.append(int(lineArray[2]))
    #dataMat: [[1.0, -0.017612, 14.053064]...[]], and every first item is 1.0
    #lineArray: the original data like: ['0.317029', '14.739025', '0'], (it's the last one)
    #labelMat: the original label like: [0, 1, ...], (the third column in the raw data)
    # print("\ndataMat: \n" + str(dataMat))
    # print("\nlineArray: \n" + str(lineArray))
    # print("\nlabelMat: \n" + str(labelMat))

    return dataMat, labelMat



#the sigmoid function
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))



#Stochastic Gradient Rise Method
def stocGradAscent(dataMatrix, classLabels):
    #get the shape of dataMatrix, m: 100(the count of data), n: 3(the number of feature), a 100*3 mat
    m, n = np.shape(dataMatrix)
    print("\nm: \n" + str(m))
    print("\nn: \n" + str(n))
    alpha = 0.01                # set step size
    weights = np.ones(n)        # the weight is initialized to 1, same as n
    # print("\nweights: \n" + str(weights))


    for i in range(m):
        # get a matrix which is 100*1, calculate the value in sigmoid function and return answer as h
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = (classLabels[i] - h)
        weights = weights + alpha*dataMatrix[i]*error

    # print("\nh: \n" + str(h))
    # print("\nerror: \n" + str(error))
    # print("\nweights: \n" + str(weights))

    return weights


# plot the image and the function
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    # print("\ndataArr: \n" + str(dataArr))
    #the number of dim of dataArr; in [], number 0 means getting the value of line; 1 means col
    n = np.shape(dataArr)[0]

    x1 = []; y1 = []
    x2 = []; y2 = []
    # cycle from 0 to 99(total 100 times), classify the 2 class points of labelMat: 1 & 0
    for i in range(n):
        if int(labelMat[i]) == 1:
            x1.append(dataArr[i,1])
            y1.append(dataArr[i,2])
        else:
            x2.append(dataArr[i,1])
            y2.append(dataArr[i,2])
    # print("\nx1: \n" + str(x1))
    # print("\ny1: \n" + str(y1))
    # print("\nx2: \n" + str(x2))
    # print("\ny2: \n" + str(y2))

    # create a new figure, and names fig
    fig = plt.figure()

    # the location of image
    # first and second 1 means total 1 line and 1 col; last 1 means the location of this image
    ax = fig.add_subplot(111)

    # mark 2 class of points: (x1,y1) use red; and x2,y2) use green
    # 's' means the size of points(default is '20'); 'c' means color
    # 'marker' means the shape of point, here 's' means square, default is 'o', means circle
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='green')

    # use the best weights to get the function of X1 & X2
    # set the range of x label: [-3.0, 3.1], step is 0.1; the actually range is [-3.0, 3.0]
    X1 = np.arange(-3.0, 3.1, 0.1)
    X2 = (-weights[0] - weights[1]*X1) / weights[2]

    # plot X1,X2 function(using default line style and color), here sigmoid is 0, means boundary
    ax.plot(X1, X2)
    # set the names of x label and y label
    plt.xlabel('X1'); plt.ylabel('X2')

    #show this image
    plt.show()



if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # print("\ndataArr: \n" + str(dataArr))
    # print("\nlabelMat: \n" + str(labelMat))

    # print '\nshape: \n', np.shape(np.array(dataArr))
    weight = stocGradAscent(np.array(dataArr), labelMat)
    print "\nweight:\n", weight
    plotBestFit(weight)