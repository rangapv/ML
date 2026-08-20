#!/usr/bin/env python3


import inspect
from gsplat.rendering import rasterization,rasterization_2dgs
from torch import Tensor, optim
#print(inspect.signature(rasterization))
print(inspect.signature(rasterization_2dgs))
print(dir(optim.Adam))
