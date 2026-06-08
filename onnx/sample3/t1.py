#!/usr/bin/env python3
#author:rangapv@yahoo.com
#10-11-24

import numpy
import onnx
from onnx import numpy_helper

# Preprocessing: create a Numpy array
numpy_array = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
print(f"Original Numpy array:\n{numpy_array}\n")

# Convert the Numpy array to a TensorProto
tensor = numpy_helper.from_array(numpy_array)
print(f"TensorProto:\n{tensor}")

# Convert the TensorProto to a Numpy array
new_array = numpy_helper.to_array(tensor)
print(f"After round trip, Numpy array:\n{new_array}\n")

# Save the TensorProto
with open("tensor.pb", "wb") as f:
    f.write(tensor.SerializeToString())

# Load a TensorProto
new_tensor = onnx.TensorProto()
with open("tensor.pb", "rb") as f:
    new_tensor.ParseFromString(f.read())
print(f"After saving and loading, new TensorProto:\n{new_tensor}")

from onnx import TensorProto, helper

# Conversion utilities for mapping attributes in ONNX IR
# The functions below are available after ONNX 1.13
np_dtype = helper.tensor_dtype_to_np_dtype(TensorProto.FLOAT)
print(f"The converted numpy dtype for {helper.tensor_dtype_to_string(TensorProto.FLOAT)} is {np_dtype}.")
storage_dtype = helper.tensor_dtype_to_storage_tensor_dtype(TensorProto.FLOAT)
print(f"The storage dtype for {helper.tensor_dtype_to_string(TensorProto.FLOAT)} is {helper.tensor_dtype_to_string(storage_dtype)}.")
field_name = helper.tensor_dtype_to_field(TensorProto.FLOAT)
print(f"The field name for {helper.tensor_dtype_to_string(TensorProto.FLOAT)} is {field_name}.")
tensor_dtype = helper.np_dtype_to_tensor_dtype(np_dtype)
print(f"The tensor data type for numpy dtype: {np_dtype} is {helper.tensor_dtype_to_string(tensor_dtype)}.")

for tensor_dtype in helper.get_all_tensor_dtypes():
    print(helper.tensor_dtype_to_string(tensor_dtype))
