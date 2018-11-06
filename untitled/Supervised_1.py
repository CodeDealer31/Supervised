from numpy import *
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
a = [1 , 2, 3, 4]

b = [3,2,0,5]
plt.plot(a,b)
c = []

class k1:
    def f(self):
        #k = 1
        a1 = a;
        for i in a:
            a1[i-1] = 1;

        #find the parameters
        ainv = mat(a1).T.I
        c = ainv*mat(b).T

        #plot
        x = np.linspace(0,5,6)
        y = range(len(x))

        for p in range(len(x)):
            y[p-1] = 1*c[0,0]

        plt.plot(x,y)

#k=2
class k2:
    def __init(self):
        pass
    def f(self):
        a2 = vstack((a,a))
        for i in range(len(a)):
            a2[0,i-1] = 1

        #find the parameters
        ainv = mat(a2).T.I
        c  = ainv*mat(b).T
        print(c)

        #plot
        x = np.linspace(0,5,)
        y = range(len(x))

        for p in range(len(x)):
            y[p-1] = 1*c[0,0]+x[p-1]*c[1,0]

        plt.plot(x,y)

#k=3
class k3:
    def f(self):
        a2 = vstack((a,a,a))
        for i in range(len(a)):
            a2[0,i-1] = 1
            a2[2,i-1] = (a[i-1]) **2
        print(a2)

        #find the parameters
        ainv = mat(a2).T.I
        c = ainv*mat(b).T
        print(c)

        #plot
        x = np.linspace(0,5,100)
        y = range(len(x))

        for p in range(len(x)):
            y[p-1]= 1*c[0,0]+x[p-1]*c[1,0]+(x[p-1]**2)*c[2,0]


        plt.plot(x, y)

#k=3
class k4:
    def f(self):
        a2 = vstack((a,a,a,a))
        for i in range(len(a)):
            a2[0,i-1] = 1
            a2[2,i-1] = (a[i-1]) **2
            a2[3, i - 1] = (a[i - 1]) ** 3
        print(a2)

        #find the parameters
        ainv = mat(a2).T.I
        c = ainv*mat(b).T
        print(c)

        #plot
        x = np.linspace(0,5,100)
        y = range(len(x))

        for p in range(len(x)):
            y[p-1]= 1*c[0,0]+x[p-1]*c[1,0]+(x[p-1]**2)*c[2,0]+(x[p-1]**3)*c[3,0]

        plt.plot(x, y)




k3().f()
plt.show()







