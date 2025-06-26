#!/usr/bin/env python3
#author: rangapv@yahoo.com
#25-06-25

import tensorflow as tf
import tensorrt as trt
import tensorflow.keras as ks
import ctypes
#import cuda.cuda as cuda
import pycuda.driver as cuda
from cuda import cuda, cudart
#import pycuda.driver as cuda
import common
import numpy as np
from typing import Optional, List, Union
import os
import sys


from PIL import Image


class ModelData(object):
    MODEL_PATH = "/usr/src/tensorrt/data/resnet50/ResNet50.onnx"
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


l1 = trt.Logger()

b1 = trt.Builder(l1)

n1 = b1.create_network()

c1 = b1.create_builder_config()

parser = trt.OnnxParser(n1, l1)

model_file = ModelData.MODEL_PATH

with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
#            return None

e1 = b1.build_engine_with_config(n1,c1)

#e1 = b1.build_serialized_network(n1, c1)

cont1 = e1.create_execution_context()

insp = e1.create_engine_inspector()

f1 = trt.LayerInformationFormat(1)

tg2 = insp.get_engine_information(f1)

print(f'tg2 is {tg2}')

ser1 = e1.serialize()

ser2 = trt.Runtime(l1)

cuda1 = ser2.deserialize_cuda_engine(ser1)

#cuda1 = ser2.deserialize_cuda_engine(e1)


#
#

class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = (cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
        print(f'dtype is {dtype}')
        print(f'nbyres os {nbytes}')
        print(f'size is {size}')
        print(f'host_meme in call is {host_mem}')
        print(f'pointer type is {pointer_type}')
        #hm1 = ctypes.pointer(host_mem)
        #print(f'the host mem pointer is {hm1}')
        ctypescast1 = ctypes.cast(host_mem[1], pointer_type)
        print(f'ctypecast1 is {ctypescast1}')
        self._host = np.ctypeslib.as_array(ctypescast1, (size,))
        #self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        dv1 = cudart.cudaMalloc(nbytes)
        self._device = (dv1[1])
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            assert self.host.dtype == np.uint8
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        (cudart.cudaFree(self.device))
        (cudart.cudaFreeHost(self.host.ctypes.data))

#

inputs = []
outputs = []
bindings = []
stream = cudart.cudaStreamCreate()
print(f'stream is {stream}')

tensor_names = [e1.get_tensor_name(i) for i in range(e1.num_io_tensors)]
print(f'tensor name sare {tensor_names}')
#idx1 = e1.getBidingIndex(tensor_names[0])
#print(f'the profile-idx is {idx1}')
profile_idx = 0 

for binding in tensor_names:
    bind1 = e1.get_tensor_shape(binding)
    bind12 = e1.get_tensor_profile_shape(binding, profile_idx)[-1]
    print(f'bind1 is {bind1}')
    print(f'bind12 is {bind12}')
    shape = e1.get_tensor_shape(binding) if profile_idx is None else e1.get_tensor_profile_shape(binding, profile_idx)[-1]
    shape_valid = np.all([s >= 0 for s in shape])
    if not shape_valid and profile_idx is None:
       raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
    size = trt.volume(shape)
    trt_type = e1.get_tensor_dtype(binding)
    print(f'tensor is {binding}')
    print(f'shape is {shape}')
    print(f'sze is {size}')
    print(f'trt_type is {trt_type}')

    if trt.nptype(trt_type):
         dtype = np.dtype(trt.nptype(trt_type))
         print(f'in f dtype is {dtype}')
         bindingMemory = HostDeviceMem(size, dtype)
    else: # no numpy support: create a byte array instead (BF16, FP8, INT4)
         size = int(size * trt_type.itemsize)
         bindingMemory = HostDeviceMem(size)


    bindings.append(int(bindingMemory.device))
    # Append to the appropriate list.
    if e1.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
         inputs.append(bindingMemory)
    else:
         outputs.append(bindingMemory)
    #return inputs, outputs, bindings, stream
    print(f'binding memeory is {bindingMemory} for {binding}')
    print(f'binding  deviceis {bindingMemory.device}for {binding}')


##

def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = (
            np.asarray(image.resize((w, h), Image.LANCZOS))
            .transpose([2, 0, 1])
            .astype(trt.nptype(ModelData.DTYPE))
            .ravel()
        )
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225
    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

test_image = "/usr/src/tensorrt/data/resnet50/binoculars.jpeg"

test_case = load_normalized_test_case(test_image, inputs[0].host)

#

num_io = e1.num_io_tensors
for i in range(num_io):
    cont1.set_tensor_address(e1.get_tensor_name(i), bindings[i])
    print(f'name is {e1.get_tensor_name(i)}')
    print(f'binding is {bindings[i]}')

add1 = cont1.set_tensor_address(e1.get_tensor_name(i), bindings[i])
print(f'add1 is {add1}')

kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
print(f'the inputs are {inputs}')
#inp = inputs[0]
#result1 = cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream[1])
result1 = (cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream) for inp in inputs )
#print(f'result1 is {result1[0]}')
#result1 = cudart.cudaMemcpyAsync(inputs[0]["device"],inputs[0]["host"],inputs[0]["nbytes"], kind, stream)

cont2 = cont1.execute_async_v3(stream_handle=stream[1])

print(f'cont2 is {cont2}')
#out = outputs[0]
print(f'outputs are {outputs}')
kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
#result2 = cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream[1])
result2 = (cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream[1]) for out in outputs )
print(f'result2 is {result2}')
    # Synchronize the stream
sync1 = cudart.cudaStreamSynchronize(stream[1])
print(f'sync1 is {sync1[0]}')
print(F'OUT-PUTS IS {outputs[0]}')
out1 = outputs[0].host
print(f'outputs or prediction is {outputs[0].host}')
print(f'out1 is {out1[0]}')
print(f'type of out1 is {type(out1[0])}')

print(f'type od out1 is {type(out1)}')
print(f'shape od out1 is {out1.shape}')

print(f'type od outputs[0] is {type(outputs[0])}')
print(f'type od outputs[0[.hosts is {type(outputs[0].host)}')

#for digit, prob in enumerate(out1):
#    print(f'{digit}: {prob:.6f}')

pred = np.argmax(out1[0])
pred1 = np.argmax(out1)

print(f'Pred1: {pred1}')
print(f'Prediction: {pred}')
print(f'Predi1: {pred1} & length of {len(out1)}')


#

labels_file = "/usr/src/tensorrt/data/resnet50/class_labels.txt" 
labels = open(labels_file, "r").read().split("\n")

pred = labels[np.argmax(out1)]
#common.free_buffers(inputs, outputs, stream)
if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
    print("Correctly recognized " + test_case + " as " + pred)
else:
    print("Incorrectly recognized " + test_case + " as " + pred)

