#!/usr/bin/env python3
#author: rangapv@yahoo.com
#20-05-25


import tensorflow as tf
import tensorflow.image as image
import tensorflow.keras as ks
import numpy as np
import os

img_path = 'ade20k.jpg'

img = ks.utils.load_img(img_path, target_size=(224, 224))
#img = load_img(img_path, target_size=(224, 224))

nparray1 = ks.utils.img_to_array(img)

nparray2 = np.array([nparray1])

pre1 = ks.applications.resnet50.preprocess_input(nparray2)

print(f'pre1 is {pre1}')

model = ks.applications.ResNet50(weights='imagenet')

y1 = model.predict(pre1)

y = ks.applications.resnet50.decode_predictions(y1)

model.save("model2.keras")

print(y)

