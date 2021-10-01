#!/usr/bin/env python
import numpy as np
from sklearn.utils import shuffle
import time as time
import matplotlib.pyplot as plt

class Sf:
    def __init__(self):
        self.x = np.array([1,2,3,4,5,6]) 
        self.y = np.array([10,20,30,40,50,60]) 
        self.a = np.array([1,2,4,5,6])


    def sh1(self):
        print("Hi")

    def sh2(self):
        
        for i in range(5):
 
        #a=np.array([1,2,4,5,6])
 
            print(f"self.a = {self.a}")
 
            np.random.shuffle(self.a)
 
            print(f"shuffled a = {self.a}\n")

    def sh3(self):
        #x = np.array([1,2,3,4,5,6])

        #y = np.array([10,20,30,40,50,60])

        print(f"x = {self.x}, y = {self.y}")

        shuffled_indices = np.random.permutation(len(self.x)) #return a permutation of the indices

        print(f"shuffled indices: {shuffled_indices}")

        x = self.x[shuffled_indices]

        y = self.y[shuffled_indices]

        print(f"shuffled x  = {x}\nshuffled y {y}")


    def sh4(self):
       # x = np.array([1,2,3,4,5,6])

        #y = np.array([10,20,30,40,50,60])

        x_shuffled, y_shuffled = shuffle(self.x,self.y)

        print(f"shuffled x = {x_shuffled}\nshuffled y={y_shuffled}")

        print(f"original x = {self.x}, original y = {self.y}")


    def sh5(self):
        x = np.random.randint(1,100, size=(3,3))

        print(f"x:\n{x}\n")

        np.random.shuffle(x)

        print(f"shuffled x:\n{x}")

#Since the shuffle from sh5 we saw the rows are merely shuffled to shuffle column transpose it and shuffle

    def sh6(self):
        x = np.random.randint(1,100, size=(3,3))

        print(f"x:\n{x}\n")

        np.random.shuffle(x.T) #shuffling transposed form of x

        print(f"column-wise shuffled x:\n{x}")


#Shuffle multdimensional array
    def sh7(self):
        x = np.random.randint(1,100, size=(4,3,3))

        print(f"x:\n{x}\n")

        np.random.shuffle(x)

        print(f"shuffled x:\n{x}")


#shuffle along any axis --same technique as in sh2 method generate permutation of indiices and then shuffle
    def sh8(self):
        x = np.random.randint(1,100, size=(4,3,3))

        print(f"x:\n{x}, shape={x.shape}\n")

        indices_1 = np.random.permutation(x.shape[1])

        x_1 = x[:,indices_1,:]

        print(f"shuffled x along axis=1:\n{x_1}, shape={x_1.shape}\n")

        indices_2 = np.random.permutation(x.shape[2])

        x_2 = x[:,:,indices_2]

        print(f"shuffled x along axis=2:\n{x_2}, shape={x_2.shape}\n")


#Sets and Dictionaries are mutable but not subscriptable. Tuples and Strings are subscriptable but not mutable.
#Lets shuffle Lists THEN
    def sh9(self):
        a = [5.4, 10.2, "hello", 9.8, 12, "world"]

        print(f"a = {a}")

        np.random.shuffle(a)

        print(f"shuffle a = {a}")


#transpose, which accepts a tuple of axis indices and reshapes the array as per the order of the axes passed

    def sh10(self):
        np.random.seed(0)

        a = np.arange(48).reshape(2,3,2,4)

        print(f"array a:\n{a}, shape={a.shape}\n")

        shuffled_dimensions = np.random.permutation(a.ndim)

        print(f"shuffled dimensions = {shuffled_dimensions}\n")

        a_shuffled = a.transpose(shuffled_dimensions)

        print(f"array a with shuffled dimensions:\n{a_shuffled}, shape={a_shuffled.shape}")


    def sh11(self):

        permutation_time_log = []

        shuffle_time_log = []

        for i in range(2,5):

            print(f"shuffling array of length 10^{i}")

            a = np.random.randint(100, size=(10**i))

            t1 = time.time()

            np.random.permutation(a)

            t2 = time.time()

            permutation_time_log.append(t2-t1)

            t1 = time.time()

            np.random.shuffle(a)

            t2 = time.time()

            shuffle_time_log.append(t2-t1)

            del a
            fig = plt.figure(figsize=(8,6))
 
            ax  = fig.add_subplot(111)
 
            ax.plot(permutation_time_log, label="permutation")
 
            ax.plot(shuffle_time_log, label="shuffle")
 
            ax.set_xlabel("length of array")
 
            ax.set_ylabel("time for shuffling(s)")
 
            ax.set_xticks(range(8))
 
            ax.set_xticklabels([f"10^{i}" for i in range(2,10)])
 
            ax.legend()
 
            plt.show()



if __name__ == '__main__':
   s1 = Sf()
   s2 =s1.sh2()
   s2 =s1.sh3()
   s2 =s1.sh4()
   s2 =s1.sh5()
   s2 =s1.sh6()
   s2 =s1.sh7()
   s2 =s1.sh8()
   s2 =s1.sh9()
   s2 =s1.sh10()
   s2 =s1.sh11()

