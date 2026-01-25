#!/usr/bin/env python3
#author:rangapv@yahoo.com
#25-01-26


import cuda.tile as ct
from math import ceil
import torch

# Type alias for compile-time constants
ConstInt = ct.Constant[int]

# Step 1: Define the kernel
@ct.kernel
def matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    # 1.1 Get block ID and map to output tile position
    # inside swizzle_2d, we access ct.bid(0) and output bidx and bidy
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    # 1.2 Calculate the number of tiles along the K dimension
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # 1.3 Initialize accumulator
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)

    # 1.4 Loop over K dimension
    for k in range(num_tiles_k):
        # Load tiles from A and B
        a = ct.load(A, index=(bidx, k), shape=(tm, tk))
        b = ct.load(B, index=(k, bidy), shape=(tk, tn))

        # Matrix multiply-accumulate
        accumulator = ct.mma(a, b, accumulator)

    # 1.5 Store result
    ct.store(C, index=(bidx, bidy), tile=accumulator)

# Step 2: Launch the kernel
def cutile_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Choose tile sizes
    tm, tn, tk = 128, 256, 64  # for float16

    # Calculate grid dimensions
    grid_x = ceil(m / tm)
    grid_y = ceil(n / tn)
    grid = (grid_x * grid_y, 1, 1)

    # Create output and launch
    C = torch.empty((m, n), device=A.device, dtype=A.dtype)
    ct.launch(stream, grid, matmul_kernel, (A, B, C, tm, tn, tk))
    return C
