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

def kernel(x,another_x,sigma = 1):
    b = (np.linalg.norm([x,another_x]))**2
    b = exp(-b/(2*(sigma**2)))
    return b

b = kernel([0,0],[0,0])
print(b)

class qs5:
    def __init__(self):
        self.sigma = [2**(7+0.5*i) for i in range(13)]
        self.gama = [2**(-40+i) for i in range(15)]



