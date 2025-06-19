#!/usr/bin/env python3
#author: rangapv@yahoo.com
#20-05-25

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

from PIL import Image

l1 = trt.Logger()

b1 = trt.Builder(l1)

print(f'builder is {b1}')

profile = b1.create_optimization_profile();
profile.set_shape("input", (1,224, 224, 3), (1,224, 224, 3), (1,224, 224, 3))
#config.add_optimization_profile(profile)


#con1 = trt.IExecutionContext()

#con1.set_input_shape("foo", (3, 150, 250))

print(f'profile is {profile}')

n1 = b1.create_network()

#n1.add_input("input", trt.float32, (1,3, -1, -1))
print(f'network is {n1}')

output = [] 
c1 = b1.create_builder_config()
c1.add_optimization_profile(profile)
print(f'config is {c1}')

#e1 = b1.build_serialized_network(n1,c1)
#e1 = b1.build_engine_with_config(n1,c1)

#cont1 = e1.create_execution_context()
#print(f'cont1 is {cont1}')

#inputs, outputs, bindings, stream = common.allocate_buffers(engine)

parser = trt.OnnxParser(n1, l1)

print(f'parser is {parser}')

#success = parser.parse_from_file("/usr/src/tensorrt/data/resnet50/ResNet50.onnx")
#success = parser.parse_from_file("./onnx1.onnx")
success = parser.parse_from_file("./resnet50.onnx")

print(f'success is {success}')

#for idx in range(parser.num_errors):
#    print(parser.get_error(idx))

#if not success:
#    pass # Error handling code here

e1 = b1.build_engine_with_config(n1,c1)

print(f'engine is {e1}')



cont1 = e1.create_execution_context()

print(f'the context is {cont1}')


ser1 = e1.serialize()

print(f'plan is {ser1}')

ser2 = trt.Runtime(l1)

print(f'runtime is {ser2}')

cuda1 = ser2.deserialize_cuda_engine(ser1)

print(f'cuda is {cuda1}')

#add1 = cont1.set_tensor_address(name, ptr)

#print(f'address is {add1}')

#stream = cuda.Stream()


#pred = cont1.execute_async_v3(cuda1)

#print(f'pred is {pred}')
"""
#stream = cuda.Stream(0)
#print (f'stream is {stream}')
print(f'len of e1 is {e1[0]}')
r2 = e1.get_device_memory_size_for_profile_v2(0)
print(f'r2 is {r2}')
r21 = e1.get_tensor_dtype
print(f'r21 is {r21}')
s1 = trt.volume(r2)
s2 = trt.nptype(r21)

r3 =  e1[1]
print(f'r3 is {r3}')


host_mem = cuda.pagelocked_empty(s1,s2)
print(f'host_mem is {host_mem}')
device_mem = cuda.mem_alloc(host_mem.nbytes)
print(f'device_mem is {device_mem}')

"""

def postprocess(data):
    num_classes = 21
    # create a color palette, selecting a color for each class
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([palette*i%255 for i in range(num_classes)]).astype("uint8")
    # plot the segmentation predictions for 21 classes in different colors
    img = Image.fromarray(data.astype('uint8'), mode='P')
    img.putpalette(colors)
    return img






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


"""
    bindings.append(int(device_mem[1]))
    if e1.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
        inputs.append({"host": h1, "device": device_mem, "nbytes": nbytes})
    else:
        outputs.append({"host": h1, "device": device_mem, "nbytes": nbytes})
"""
#print(f'inpit0  {inputs[0]["device"]} {inputs[0]["host"]} {inputs[0]["nbytes"]}')
#print(f'outt0  {outputs[0]["device"]} {outputs[0]["host"]} {outputs[0]["nbytes"]}')
print(f'inputs is {inputs} and outputs is {outputs} and bindigns is {bindings}')

def postprocess(data):
    num_classes = 21
    # create a color palette, selecting a color for each class
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([palette*i%255 for i in range(num_classes)]).astype("uint8")
    # plot the segmentation predictions for 21 classes in different colors
    img = Image.fromarray(data.astype('uint8'), mode='P')
    img.putpalette(colors)
    return img


"""

"""

num_io = e1.num_io_tensors
for i in range(num_io):
    cont1.set_tensor_address(e1.get_tensor_name(i), bindings[i])

print(f'name is {e1.get_tensor_name(i)}')
print(f'binding is {bindings[i]}')
add1 = cont1.set_tensor_address(e1.get_tensor_name(i), bindings[i])
print(f'add1 is {add1}')

kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice

inp = inputs[0]
result1 = cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream[1])
#result1 = (cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream) for inp in inputs )
print(f'result1 is {result1[0]}')
#result1 = cudart.cudaMemcpyAsync(inputs[0]["device"],inputs[0]["host"],inputs[0]["nbytes"], kind, stream)
    
cont2 = cont1.execute_async_v3(stream_handle=stream[1])

print(f'cont2 is {cont2}')
out = outputs[0]
kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
result2 = cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream[1])
print(f'result2 is {result2[0]}')
    # Synchronize the stream
sync1 = cudart.cudaStreamSynchronize(stream[1])
print(f'sync1 is {sync1[0]}')
print(F'OUT-PUTS IS {outputs[0]}')
out1 = outputs[0].host
print(f'outputs or prediction is {outputs[0].host}')

#for digit, prob in enumerate(out1):
#    print(f'{digit}: {prob:.6f}')
pred = np.argmax(out1)


print(f'Prediction: {pred}')

j1 = ks.applications.resnet50.decode_predictions(out)
print(f'tensorRT prediction is {j1}')

input_file  = "input.ppm"
output_file = "output.ppm"

image_height = 10 
image_width = 100 
with postprocess(np.reshape(out1, (image_height, image_width))) as img:
        print("Writing output image to file {}".format(output_file))
        img.convert('RGB').save(output_file, "PPM")



labels_file = "/usr/src/tensorrt/data/resnet50/class_labels.txt"


labels = open(labels_file, "r").read().split("\n")
pred = labels[np.argmax(out1)]
print(f'predout is {pred}')
print(f'predout-str is {str(pred)}')
#with postprocess(np.reshape(outputs[0].host, (224, 224))) as img:
#        print("Writing output image to file {}".format(output_file))
#        img.convert('RGB').save(output_file, "PPM")
