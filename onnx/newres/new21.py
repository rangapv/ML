#!/usr/bin/env python3
#author: rangapv@yahoo.com
#20-05-25

import tensorflow as tf
import tensorflow.image as image
import tensorflow.keras as ks
import numpy as np
import os
import tf2onnx
import onnxruntime as rt
import onnx
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

img_path = './beach.jpg'

img = ks.utils.load_img(img_path, target_size=(224, 224))

nparray1 = ks.utils.img_to_array(img)

nparray2 = np.array([nparray1])

pre1 = ks.applications.resnet50.preprocess_input(nparray2)

model = ks.applications.resnet50.ResNet50(weights='imagenet')

y1 = model.predict(pre1)
print(f'the type of ys is {type(y1)}')
print(f'the shape of y1 is {y1.shape}')
print(f'printed y1 is {y1}')

y = ks.applications.resnet50.decode_predictions(y1)


model.save("model2.h5")
#model.save("model2.keras")
#now the onnx part

spec = tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input")

output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=[spec], opset=13, output_path=output_path)

onnx.save_model(model_proto,"onnx1.onnx")

output_names = [n.name for n in model_proto.graph.output]

providers = ['CUDAExecutionProvider']

m = rt.InferenceSession(output_path, providers=providers)
print(f'the tpe of m is {m}')
print(f'm is {m}')
onnx_pred = m.run(output_names, {"input": pre1})
print(f'the type of onnx_pred is  {type(onnx_pred)}')
print(f'the shape of onnx_pred is  {onnx_pred}')
print(f'printed onnx_pred is {onnx_pred}')

print('ONNX Predicted:', ks.applications.resnet50.decode_predictions(onnx_pred[0], top=3)[0])
print(f'tensor predicted is {y}')

np.testing.assert_allclose(y1, onnx_pred[0], rtol=1e-0)

#now the tensoRT inference Part



