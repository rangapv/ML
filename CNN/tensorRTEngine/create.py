#!/usr/bin/env python3
#author: rangapv@yahoo.com
#17-05-25


import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(logger)

network = builder.create_network(1)

config = builder.create_builder_config()


engine = builder.build_engine_with_config(network,config)

cacmem = builder.build_serialized_network(network,config)

print(logger)
print(builder)
print(network)
print(config)
print(engine)
print(cacmem)
