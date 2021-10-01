#!/usr/bin/env python
import numpy as np


class Sf:
    def __init__(self):
        self.x = 6
        self.y = 5
        self.a = np.array([1,2,4,5,6])


    def sh1(self):
        print("Hi")

    def sh2(self):
        
        for i in range(5):
 
        #a=np.array([1,2,4,5,6])
 
            print(f"self.a = {self.a}")
 
            np.random.shuffle(self.a)
 
            print(f"shuffled self.a = {self.a}\n")




if __name__ == '__main__':
   s1 = Sf()
   s2 =s1.sh2()
