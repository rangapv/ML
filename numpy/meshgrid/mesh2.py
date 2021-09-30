#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt



class Example:

    def __init__(self):
        self.X = np.array([1,2,3,4,5])

        self.Y = np.array([11, 12, 13, 14, 15])
        print("X is {}".format(self.X))


    def func1(self):

        xx, yy = np.meshgrid(self.X,self.Y)

        print("X coordinates:\n{}\n".format(xx))

        print("Y coordinates:\n{}".format(yy))

   
    def func2(self):
        xx, yy = np.meshgrid(self.X,self.Y, sparse=True)

        print("X coordinates:\n{}\n".format(xx))

        print("Y coordinates:\n{}".format(yy))

    def func3(self):
        R = np.linspace(1,5,10)

        THETA = np.linspace(0, np.pi, 45)

        radii, thetas = np.meshgrid(R,THETA)

        print("R:{}".format(R.shape))

        print("THETA:{}".format(THETA.shape))

        print("meshgrid radii:{}".format(radii.shape))

        print("mehgrid thetas:{}".format(thetas.shape))


    def func4(self):
        #meshgrid with matrix indexing
        i = np.array([1,2,3,4,5]) #rows

        j = np.array([11, 12, 13, 14, 15]) #columns

        ii, jj = np.meshgrid(i,j, indexing='ij')

        print("row indices:\n{}\n".format(ii))

        print("column indices:\n{}".format(jj))


    def func5(self):

        #Visualizing the meshgrid

        X = np.linspace(1,15,15)

        Y = np.linspace(20,30,10)

        xx, yy = np.meshgrid(X,Y)

        fig = plt.figure()

        ax = fig.add_subplot(111)

        ax.plot(xx, yy, ls="None", marker=".")

        plt.show()
        
        
        
    def func6(self):
        #Flipping it

        fig = plt.figure()

        ax = fig.add_subplot(111)

        ax.plot(xx[:,::-1], yy[:,::-1], ls="None", marker=".")

        plt.show()   
        
        
    def func7(self):
  
        #meshgrid with Numpy matrices

        np.random.seed(42)

        a = np.random.randint(1,5, (2,2))

        b = np.random.randint(6,10, (3,3))

        print("a:\n{}\n".format(a))

        print("b:\n{}".format(b))

        xx, yy = np.meshgrid(a,b)

        print("xx:\n{}".format(xx))

        print("shape of xx:{}\n".format(xx.shape))

        print("yy:\n{}".format(yy))

        print("shape of yy:{}\n".format(yy.shape))



    def func71(self):
        R = np.linspace(1,5,10)

        THETA = np.linspace(0, np.pi, 45)

        radii, thetas = np.meshgrid(R,THETA)

        print("R:{}".format(R.shape))

        print("THETA:{}".format(THETA.shape))

        print("meshgrid radii:{}".format(radii.shape))

        print("mehgrid thetas:{}".format(thetas.shape))
        return radii,thetas

    def func8(self,radii,thetas):
        #meshgrid with pyplot


        ax = plt.subplot(111, polar=True)

        ax.plot(thetas, radii, marker='.', ls='none')

        plt.show()   


    def func9(self):
        xx, yy = np.meshgrid(a.ravel(),b.ravel()) #passing flattened arrays
 
        print("xx:\n{}".format(xx))
 
        print("shape of xx:{}\n".format(xx.shape))
 
        print("yy:\n{}".format(yy))
 
        print("shape of yy:{}\n".format(yy.shape))
        
        
        
    def func10(self):
        X = np.linspace(1,4,4)
 
        Y = np.linspace(6,8, 3)
 
        Z = np.linspace(12,15,4)
 
        xx, yy, zz = np.meshgrid(X,Y,Z)
 
        print(xx.shape, yy.shape, zz.shape) 
        
        
        #    def func11(self):
#    def func12(self):
#    def func13(self):
#    def func14(self):
#    def func15(self):
#    def func16(self):
#    def func17(self):
#    def func18(self):
#    def func19(self):
#    def func20(self):
#    def func21(self):
#    def func22(self):
#    def func23(self):

if __name__== '__main__':
    v = Example()
    v1 = v.func1()
    v1 = v.func2()
    v1 = v.func3()
    v1 = v.func4()
   # v1 = v.func5()
   # v1 = v.func6()
    v1 = v.func7()
    r,t = v.func71()
    v1 = v.func8(r,t)
