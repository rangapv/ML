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

img_path = './tests/car.JPEG'

img = ks.utils.load_img(img_path, target_size=(224, 224))

nparray1 = ks.utils.img_to_array(img)

nparray2 = np.array([nparray1])

pre1 = ks.applications.resnet50.preprocess_input(nparray2)

print(f'pre1 is {pre1}')

model = ks.applications.ResNet50(weights='imagenet')

y1 = model.predict(pre1)

y = ks.applications.resnet50.decode_predictions(y1)

model.save("model2.keras")

print(y)

#now the onnx part

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
#output_path = "newrt.onnx"

output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)

onnx.save_model(model_proto,"onnx1.onnx")

#output_path = "onnx1.onnx"

output_names = [n.name for n in model_proto.graph.output]

print(f'output name s is {output_names}')

providers = ['CUDAExecutionProvider']

m = rt.InferenceSession(output_path, providers=providers)

onnx_pred = m.run(output_names, {"input": pre1})

# make sure ONNX and keras have the same results
np.testing.assert_allclose(y1, onnx_pred[0], rtol=1e-5)
