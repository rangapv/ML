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
        return a,b


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


    def func9(self,a,b):
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
        
        
    def func11(self):
        #metrix indexing
        i = np.array([1,2,3,4,5]) #rows

        j = np.array([11, 12, 13, 14, 15]) #columns

        ii, jj = np.meshgrid(i,j, indexing='ij')

        print("row indices:\n{}\n".format(ii))

        print("column indices:\n{}".format(jj))
         
         
         
         
         
    def func12(self):
        #3-dimensional meshgrid
        X = np.linspace(1,4,4)
 
        Y = np.linspace(6,8, 3)
 
        Z = np.linspace(12,15,4)
 
        xx, yy, zz = np.meshgrid(X,Y,Z)
 
        print(xx.shape, yy.shape, zz.shape)
         
        return xx,yy,zz 
         
         
    def func13(self,xx,yy,zz):
        #visulaize 3d
        fig = plt.figure()
 
        ax = fig.add_subplot(111, projection='3d')
 
        ax.scatter(xx, yy, zz)
 
        ax.set_zlim(12, 15)
 
        plt.show()        
         
         
         
    def func14(self,xx,yy):
        #3D surface plot using Numpy
        X = np.linspace(-20,20,100)
 
        Y = np.linspace(-20,20,100)
 
        X, Y = np.meshgrid(X,Y)
 
        Z = 4*xx**2 + yy**2
 
        fig = plt.figure()
 
        ax = fig.add_subplot(111, projection='3d')
 
        ax.plot_surface(X, Y, Z, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
 
        plt.show()        
         
         
    def func15(self):
        X = np.random.randn(100000)
 
        Y = np.random.randn(100000)
 
        xx,yy =  np.meshgrid(X,Y)        
         
         
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
    a,b = v.func7()
    r,t = v.func71()
    v1 = v.func8(r,t)
    v1 = v.func9(a.b)
    xx,yy,zz = v.func12()
    v1 = v.func13(xx,yy,zz)
    v1 = v.func14(xx,yy)
    v1 = v.func15()
