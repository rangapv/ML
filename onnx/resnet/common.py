
import argparse
import os

import tensorflow as tf
import tensorrt as trt
import tensorflow.keras as ks
import ctypes
#import cuda.cuda as cuda
import pycuda.driver as cuda
from cuda import cuda, cudart
#import pycuda.driver as cuda
import numpy as np
from typing import Optional, List, Union
import os
import sys
import urllib
import urllib.request


try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def GiB(val):
    return val * 1 << 30



def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
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
        #self._host = np.ctypeslib.as_array(ctypescast1, (size,))
        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        dv1 = cudart.cudaMalloc(nbytes)
        #self._device = (dv1[1])
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
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
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))

#
def allocate_buffers(engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
 inputs = []
 outputs = []
 bindings = []
 stream = cuda_call(cudart.cudaStreamCreate())
 print(f'stream is {stream}')

 tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
 print(f'tensor name sare {tensor_names}')
#idx1 = e1.getBidingIndex(tensor_names[0])
#print(f'the profile-idx is {idx1}')
 #profile_idx = 0

 for binding in tensor_names:
    bind1 = engine.get_tensor_shape(binding)
    bind12 = engine.get_tensor_profile_shape(binding, profile_idx)[-1]
    print(f'bind1 is {bind1}')
    print(f'bind12 is {bind12}')
    shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
    shape_valid = np.all([s >= 0 for s in shape])
    if not shape_valid and profile_idx is None:
       raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
    size = trt.volume(shape)
    trt_type = engine.get_tensor_dtype(binding)
    print(f'tensor is {binding}')
    print(f'shape is {shape}')
    print(f'sze is {size}')
    print(f'trt_type is {trt_type}')


    try:
        dtype = np.dtype(trt.nptype(trt_type))
        bindingMemory = HostDeviceMem(size, dtype)
    except TypeError: # no numpy support: create a byte array instead (BF16, FP8, INT4)
        size = int(size * trt_type.itemsize)
        bindingMemory = HostDeviceMem(size)

    # Append the device buffer to device bindings.
    bindings.append(int(bindingMemory.device))

    # Append to the appropriate list.
    if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
        inputs.append(bindingMemory)
    else:
        outputs.append(bindingMemory)
 return inputs, outputs, bindings, stream



def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

    # Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


def _do_inference_base(inputs, outputs, stream, execute_async_func):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
    # Run inference.
    execute_async_func()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, engine, bindings, inputs, outputs, stream):
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream)
    # Setup context tensor address.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream, execute_async_func)
