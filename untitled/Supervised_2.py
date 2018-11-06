from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Generatedata:
    def __init__(self):
        pass
    def f(self,sample_number=30):

        # data generation
        sampleNo = sample_number
        mu = 0
        sigma = 0.07
        np.random.seed(0)
        s = np.random.normal(mu, sigma, sampleNo)

        b = np.linspace(0,1,1000)
        b2 = random.sample(list(b),sampleNo)  # b2 is out sample point in x

        realyforb2 = [(math.sin(2*pi*i))**2 for i in b2]  # this is the y value of out sample point
        yforb2 = realyforb2+s   # add some errors

        # plt.figure()
        # plt.plot(b2,yforb2,'*')
        return b2,yforb2,realyforb2

    def fdraw(self):
        plt.show()


#superimposing
class k2:
    def __init__(self):
        self.x = np.linspace(0, 1, 100)
        self.y = np.linspace(0, 0, 100)

    def findpara(self, para, xpara):
        initialxvalues = xpara
        k = para  #dim
        a2 = vstack(initialxvalues for i in range(k))
        for i in range(len(initialxvalues)):
            for kp in range(k):
                a2[kp, i-1] = (initialxvalues[i-1])**(kp)

        # find the parameters
        ainv = mat(a2).T.I
        c = ainv * mat(yforb2).T
        return c

    def evaluate(self, para,objective ):

        c = para
        k = len(c)
        # plot
        # x = np.linspace(0, 1, 100)
        # y = np.linspace(0,0,100)

        x = self.x
        y = self.y

        b2vector = objective
        estimates = np.linspace(0,0,len(b2vector))

        for p in range(len(x)):
            for kp in range(k):
                    y[p - 1] = y[p - 1] + (x[p - 1] ** kp) * c[kp, 0]

        for p in range(len(b2vector)):
            for kp in range(k):
                estimates[p - 1] = estimates[p - 1] + (b2vector[p - 1] ** kp) * c[kp, 0]

        #plt.plot(x, y,'b^',b2,yforb2,'r*')

        self.x = x
        self.y = y
        return estimates

    def fdraw(self):
        plt.show()

    def MSE(self,dim,b2):

        trainpara= self.findpara(dim, b2)
        estimatesb2 = self.evaluate(trainpara, b2)

        x_MSE = b2
        y_MSE = yforb2

        MSEvector = [(estimatesb2[i]-y_MSE[i])**2 for i in range(len(x_MSE))]
        MSE = sum(MSEvector)/len(b2)
        return MSE

    def plotlog(self, b2):

        mseinx = []
        for ii in range(19):
            print(ii)
            MSE1 = self.MSE(ii + 1, b2)
            mseinx.append(MSE1)

        logmseinx = [math.log(i) for i in mseinx]
        xaxix = range(19)
        print(logmseinx)
        print (xaxix)
        plt.plot(xaxix, logmseinx,'r*-')

    def testset(self,dim,basis):
        [b3, yforb3,realyforb3] = Generatedata().f(1000)
        c = self.findpara(dim,basis)
        estimates = self.evaluate(c, b3)

        SSE = [(estimates[i]-realyforb3[i])**2 for i in range(len(b3))]
        MSE = sum(SSE)/len(b3)
        return MSE

    def plotlogtestsets(self,basis):

        mseinx = []
        for ii in range(19):
            print(ii)
            MSE1 = self.testset(ii + 1,basis)
            mseinx.append(MSE1)

        logmseinx = [math.log(i) for i in mseinx]
        xaxix = range(19)
        print(logmseinx)
        print(xaxix)
        plt.plot(xaxix, logmseinx, 'r*-')



    def smoothd2(self):

        mseinx = []
        for ii in range(19):
            print(ii)
            MSE1 = 0
            for pp in range(100):
                [b2, yforb2, realyforb2] = Generatedata().f()
                MSE1 = MSE1+self.testset(ii + 1, b2)
            MSE1 = MSE1/100
            mseinx.append(MSE1)

        logmseinx = [math.log(i) for i in mseinx]
        xaxix = range(19)
        print(logmseinx)
        print(xaxix)
        plt.plot(xaxix, logmseinx, 'r*-')


    def smoothd1(self):

        mseinx = []
        for ii in range(19):
            print(ii)
            MSE1 = 0
            for pp in range(100):
                [b2, yforb2, realyforb2] = Generatedata().f()
                MSE1 = MSE1+self.MSE(ii + 1, b2)
            MSE1 = MSE1/100
            mseinx.append(MSE1)

        logmseinx = [math.log(i) for i in mseinx]
        xaxix = range(19)
        print(logmseinx)
        print(xaxix)
        plt.plot(xaxix, logmseinx, 'r*-')


[b2,yforb2,realyforb2] = Generatedata().f()





# print(b2)


# Generatedata().f()
# mseinx = []
# for ii in range(18):
#     Generatedata().f()
#     MSE1 = k2().MSE(ii+1)
#     mseinx.append(MSE1)
# print (mseinx)

k2().smoothd1()
plt.show()




