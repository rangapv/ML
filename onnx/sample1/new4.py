#!/usr/bin/env python3
#author: rangapv@yahoo.com
#20-05-25

import tensorflow as tf
import tensorrt as trt

#import cuda.cuda as cuda
import pycuda.driver as cuda
from cuda import cuda, cudart
#import pycuda.driver as cuda
import common
import numpy as np

l1 = trt.Logger()

b1 = trt.Builder(l1)

print(f'builder is {b1}')

profile = b1.create_optimization_profile();
profile.set_shape("input", (3,224, 224, 3), (3,224, 224, 3), (3,224, 224, 3))
#config.add_optimization_profile(profile)


#con1 = trt.IExecutionContext()

#con1.set_input_shape("foo", (3, 150, 250))

#print(f'profile is {profile}')

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

success = parser.parse_from_file("./onnx1.onnx")

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
inputs = []
outputs = []
bindings = []
stream = cudart.cudaStreamCreate()

tensor_names = [e1.get_tensor_name(i) for i in range(e1.num_io_tensors)]
print(f'tensor name sare {tensor_names}')
profile_idx = 0 

for binding in tensor_names:
    shape = e1.get_tensor_shape(binding) if profile_idx is None else e1.get_tensor_profile_shape(binding, profile_idx)[-1]
    shape_valid = np.all([s >= 0 for s in shape])
    if not shape_valid and profile_idx is None:
       raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
    size = trt.volume(shape)
    trt_type = e1.get_tensor_dtype(binding)
    print(f'shape is {shape}')
    print(f'sze is {size}')
    print(f'trt_type is {trt_type}')
   
    #dtype = dtype or np.dtype(np.uint8)
    dtype = trt.nptype(trt_type)
    print(f'dtype is {dtype}')
    t2 = np.dtype(dtype).itemsize
    print(f't2 is {t2}')
    nbytes = size * t2 
    host_mem = cudart.cudaMallocHost(nbytes)
    #pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
    device_mem = cudart.cudaMalloc(nbytes)
    print(f'hostmem is {host_mem}')
    print(f'devicemem is {device_mem}')
    # Allocate host and device buffers

    bindings.append(int(device_mem[1]))
    if e1.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
        inputs.append({"host": host_mem, "device": device_mem, "nbytes": int(nbytes)})
    else:
        outputs.append({"host": host_mem, "device": device_mem, "nbytes": int(nbytes)})

    
kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
cudart.cudaMemcpyAsync(inputs[0]["device"], inputs[0]["host"], size,kind, stream)
    
cont2 = cont1.execute_async_v3(stream_handle=stream)
    
kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
for out in outputs:
   cudart.cudaMemcpyAsync(out['host'], out['device'], out['nbytes'], kind, stream)
    # Synchronize the stream
cudart.cudaStreamSynchronize(stream)

print(f'infer is {outputs[0]}')

#size = trt.volume(e1.get_binding_shape(binding)) * e1.max_batch_size
#dtype = trt.nptype(e1.get_binding_dtype(binding))

#print(f'size is {size}')
#print(f'dtype is {dtype}')

#[output] = common.do_inference(cont1, stream=stream)
