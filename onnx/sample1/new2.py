#!/usr/bin/env python3
#author: rangapv@yahoo.com
#20-05-25

import tensorflow as tf
import tensorrt as trt

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


