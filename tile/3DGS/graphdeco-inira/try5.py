#!/usr/bin/env python3
#author:rangapv@yahoo.com
#30-8-2026


import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from PIL import Image
import numpy as np

# 1. Set up basic image dimensions
width = 800
height = 600

# 2. Create 3D Gaussian properties as LEARNABLE parameters
num_gaussians = 2000  # more points gives the optimizer more to work with

means3D = (torch.rand((num_gaussians, 3), device="cuda") * 2 - 1)  # random init in [-1, 1]
means3D.requires_grad_(True)

shs = torch.zeros((num_gaussians, 16, 3), device="cuda")
shs[:, 0, :] = 0.5  # DC term ~ gray start; SH degree 0 controls base color
shs.requires_grad_(True)

opacity = torch.ones((num_gaussians, 1), device="cuda") * 0.5
opacity.requires_grad_(True)

scales = torch.ones((num_gaussians, 3), device="cuda") * 0.05
scales.requires_grad_(True)

rotations = torch.zeros((num_gaussians, 4), device="cuda")
rotations[:, 0] = 1.0
rotations.requires_grad_(True)

#means2D = torch.zeros((num_gaussians, 2), device="cuda", requires_grad=True)
means2D = torch.zeros_like(means3D, requires_grad=True)
# 3. Set up camera matrices (real projection, not identity)

def getWorld2View(R, t):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

R = torch.eye(3)
T = torch.tensor([0.0, 0.0, 3.0])

fovx = 2 * math.atan(0.5)
fovy = 2 * math.atan(0.5)

world_view_transform = getWorld2View(R, T).transpose(0, 1).cuda()
projection_matrix = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).cuda()

projmatrix = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
viewmatrix = world_view_transform
cam_pos = viewmatrix.inverse()[3, :3]


# 4. Ground-truth tricolor image (top=red, middle=white, bottom=green)
gt_image = torch.ones((height, width, 3), device="cuda") * 1.0
gt_image[: height // 3, :, :] = torch.tensor([1.0, 0.0, 0.0], device="cuda")
gt_image[2 * height // 3 :, :, :] = torch.tensor([0.0, 1.0, 0.0], device="cuda")
gt_image_chw = gt_image.permute(2, 0, 1).contiguous()  # [3, H, W], matches rasterizer output

# 5. Optimizer over all learnable Gaussian properties
optimizer = torch.optim.Adam(
    [means3D, shs, opacity, scales, rotations],
    lr=0.01,
)

# 6. Training loop
num_iters = 500
for it in range(num_iters):
    settings = GaussianRasterizationSettings(
        antialiasing=True,
        image_height=height,
        image_width=width,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),  # black bg
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=3,
        campos=cam_pos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    rendered_image, radii, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=torch.sigmoid(opacity),   # keep opacity in [0,1]
        scales=torch.abs(scales),           # keep scales positive
        rotations=torch.nn.functional.normalize(rotations, dim=-1),  # valid quaternion
        cov3D_precomp=None,
    )

    loss = torch.nn.functional.mse_loss(rendered_image, gt_image_chw)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 50 == 0 or it == num_iters - 1:
        print(f"iter {it:4d}  loss {loss.item():.6f}")

torch.cuda.synchronize()

# 7. Save final rendered image and ground truth for comparison
arr = rendered_image.detach().cpu().numpy()       # [3, 600, 800]
arr = np.transpose(arr, (1, 2, 0))                  # -> [600, 800, 3]
arr = (arr * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(arr).save("rendered_output.png")

gt_arr = (gt_image.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(gt_arr).save("gt_tricolor.png")

print("Saved rendered_output.png and gt_tricolor.png")
