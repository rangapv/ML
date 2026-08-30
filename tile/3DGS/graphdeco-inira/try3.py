#!/usr/bin/env python3
#author:rangapv@yahoo.com
#30-08-2026

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from PIL import Image
import numpy as np
from torch import Tensor, optim


# 1. Set up basic image dimensions
width = 800
height = 600

# 2. Create dummy 3D Gaussian properties (Move to GPU)
means3D = torch.zeros((100, 3), device="cuda")
shs = torch.zeros((100, 16, 3), device="cuda")  # Spherical Harmonics colors
opacity = torch.ones((100, 1), device="cuda")
scales = torch.ones((100, 3), device="cuda")
rotations = torch.zeros((100, 4), device="cuda")  # Quaternions
rotations[:, 0] = 1.0

# 3. Set up camera matrices (Dummy 4x4 identities for example)
viewmatrix = torch.eye(4, device="cuda")
projmatrix = torch.eye(4, device="cuda")
cam_pos = torch.zeros(3, device="cuda")

#gt_image: Tensor
#gt_image = []
#device = torch.device("cuda:0")
#gt_image.to(device=device)


gt_image = torch.ones((height, width, 3)) * 1.0

        # make top left and bottom right red, blue

        # top third: red
gt_image[: height // 3, :, :] = torch.tensor([1.0, 0.0, 0.0])

        # middle third stays white (no need to set it, already white)

        # bottom third: green
gt_image[2 * height // 3 :, :, :] = torch.tensor([0.0, 1.0, 0.0])


# 4. Configure the rasterizer settings
settings = GaussianRasterizationSettings(
    antialiasing=True,
    image_height=height,
    image_width=width,
    tanfovx=0.5,
    tanfovy=0.5,
    bg=gt_image, # Black background
    #bg=torch.tensor([1, 0, 0], dtype=torch.float32, device="cuda"), # Black background
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=3,
    campos=cam_pos,
    prefiltered=False,
    debug=False
)

# 5. Initialize rasterizer and render
rasterizer = GaussianRasterizer(raster_settings=settings)
rendered_image, radii, _ = rasterizer(
    means3D=means3D,
    means2D=torch.zeros((100, 2), device="cuda"), # Keeps track of gradients
    shs=shs,
    colors_precomp=None,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None
)

# 'rendered_image' is a PyTorch tensor ready for backpropagation!
#print(rendered_image) # Output layout: [3, height, width]


#arr = rendered_image.detach().cpu().numpy()          # [3, 600, 800]
#arr = np.transpose(arr, (1, 2, 0))            # -> [600, 800, 3]
#arr = (arr * 255).clip(0, 255).astype(np.uint8)
#Image.fromarray(arr).save("output.png")

img_np = (rendered_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(img_np).save("tricolor.png")



#print(rendered_image.shape) # Output layout: [3, height, width]
print(rendered_image.shape) # Output layout: [3, height, width]
