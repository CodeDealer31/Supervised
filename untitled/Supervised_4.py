from numpy import *
import scipy.io as scio
import pandas as pd
import random
import numpy as np

data_path="boston.mat"
data = scio.loadmat(data_path)['boston']

class Generaterandomdata:
    def __init__(self):
        self.data = data

    def yshuffle(self):
        y_values = self.data[:,13]
        random.shuffle(y_values)

        train_set = y_values[0:337]

        test_set = y_values[337:506]

        return train_set,test_set

    def attributeshuffle(self,col):
        data2 = self.data
        dice = list(range(506))
        random.shuffle(dice)

        train_set = []
        test_set = []
        b=1
        for i in dice:

            if b<=337:
                train_set.append(list(data2[i]))
            else:
                test_set.append(list(data2[i]))
            b = b + 1
        train_set = hstack((mat(train_set)[:,col],mat(train_set)[:,13]))
        test_set = hstack((mat(test_set)[:, col], mat(test_set)[:, 13]))
        return train_set,test_set

    def shuffle(self):
        data2 = self.data
        dice = list(range(506))
        random.shuffle(dice)

        train_set = []
        test_set = []
        b = 1
        for i in dice:

            if b <= 337:
                train_set.append(list(data2[i]))
            else:
                test_set.append(list(data2[i]))
            b = b + 1
        train_set = hstack((mat(train_set), mat(train_set)[:, 13]))
        test_set = hstack((mat(test_set), mat(test_set)[:, 13]))
        return train_set, test_set




class Learner:

    def naiveregression(self, trainset, testset):

        constant_funciton = np.linspace(1, 1, len(trainset))

        inv = mat(constant_funciton).T.I
        c = inv*mat(trainset).T
        MSE_train = sum([(trainset[i]-c[0,0])**2 for i in range(len(trainset))])/len(trainset)
        MSE_test = sum([(testset[i]-c[0,0])**2 for i in range(len(testset))])/len(testset)
        return MSE_train,MSE_test

    def averagenaiveregression(self,times = 20):
        MSE_train_set = 0
        MSE_test_set = 0
        for i in range(times):
            print(i)
            [tr, te] = Generaterandomdata().yshuffle()
            [MSEtr,MSEte] = self.naiveregression(tr,te)
            MSE_test_set = MSE_test_set+MSEte/times
            MSE_train_set = MSE_train_set+MSEtr/times

        return MSE_train_set,MSE_test_set


## after training, constant fucntion ouput mean value of training set of y.

    def single_linear_reg(self,trainset, testset):

        constant_funciton = mat(np.linspace(1, 1, len(trainset))).T
        print(len(constant_funciton))

        train_feature = trainset[:, 0]
        train_y_feature = trainset[:, 1]

        test_feature = testset[:,0]

        test_y_feature = testset[:,1]


        feature_vec = hstack((constant_funciton, train_feature))

        inv = feature_vec.I
        c = (inv * train_y_feature).T
        print(c)

        MSE_train = sum([(train_y_feature[i, 0] - c[0, 0] - c[0, 1] * train_feature[i, 0]) ** 2 for i in
                         range(len(train_feature))]) / len(train_feature)

        MSE_test = sum([(test_y_feature[i, 0] - c[0, 0] - c[0, 1] * test_feature[i, 0]) ** 2 for i in
                        range(len(test_feature))]) / len(test_feature)


        print(MSE_train)
        print(MSE_test)
        return MSE_train,MSE_test

    def averagesinglereg(self,times = 20):
        MSE_train_set = 0
        MSE_test_set = 0
        for i in range(times):
            print(i)
            [tr, te] = Generaterandomdata().attributeshuffle(9)
            [MSEtr,MSEte] = self.single_linear_reg(tr,te)

            MSE_test_set = MSE_test_set + (MSEte/times)
            MSE_train_set = MSE_train_set+ (MSEtr/times)

        return MSE_train_set,MSE_test_set

    def all_reg(self,trainset,testset):

        constant_funciton = mat(np.linspace(1, 1, len(trainset))).T

        train_feature = trainset[:, 0:13]
        train_y_feature = trainset[:, 14]

        test_feature = testset[:, 0:13]

        test_y_feature = testset[:, 14]

        feature_vec = hstack((constant_funciton, train_feature))

        inv = feature_vec.I
        c = (inv * train_y_feature).T
        print(c)

        MSE_train = sum([(train_y_feature[i, 0] - c[0, 0] - c[0, 1] * train_feature[i, 0]) ** 2 for i in
                         range(len(train_feature))]) / len(train_feature)

        MSE_test = sum([(test_y_feature[i, 0] - c[0, 0] - c[0, 1] * test_feature[i, 0]) ** 2 for i in
                        range(len(test_feature))]) / len(test_feature)


[tr,te]=Generaterandomdata().shuffle()
Learner().all_reg(tr,te)

# Learner().averagesinglereg()

