#!/usr/bin/env python3
#author:rangapv@yahoo.com
#30-08-2026

#We are going to generate a GIF and then reverse the order in the same picture! Wow!

import os
import glob
import math
import torch
import numpy as np
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# ----------------------------------------------------------------------
# 1. Basic image dimensions
# ----------------------------------------------------------------------
width = 800
height = 600

# ----------------------------------------------------------------------
# 2. Learnable 3D Gaussian properties
# ----------------------------------------------------------------------
num_gaussians = 2000

means3D = (torch.rand((num_gaussians, 3), device="cuda") * 2 - 1)  # random init in [-1, 1]
means3D.requires_grad_(True)

shs = torch.zeros((num_gaussians, 16, 3), device="cuda")
shs[:, 0, :] = 0.5  # degree-0 (DC) term ~ gray start
shs.requires_grad_(True)

opacity = torch.ones((num_gaussians, 1), device="cuda") * 0.5
opacity.requires_grad_(True)

scales = torch.ones((num_gaussians, 3), device="cuda") * 0.05
scales.requires_grad_(True)

rotations = torch.zeros((num_gaussians, 4), device="cuda")
rotations[:, 0] = 1.0
rotations.requires_grad_(True)

# gradient accumulator buffer -- must match means3D shape, i.e. [N, 3], not [N, 2]
means2D = torch.zeros_like(means3D, requires_grad=True)

# ----------------------------------------------------------------------
# 3. Camera matrices (correct row-vector / transposed convention)
# ----------------------------------------------------------------------
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
T = torch.tensor([0.0, 0.0, 3.0])  # camera 3 units back from origin along +z

fovx = 2 * math.atan(0.5)  # matches tanfovx=0.5
fovy = 2 * math.atan(0.5)  # matches tanfovy=0.5

world_view_transform = getWorld2View(R, T).transpose(0, 1).cuda()
projection_matrix = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).cuda()

projmatrix = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
viewmatrix = world_view_transform
cam_pos = viewmatrix.inverse()[3, :3]

# ----------------------------------------------------------------------
# 4. Ground-truth images: normal tricolor, and its vertical color-reverse
# ----------------------------------------------------------------------
# Normal: top=red, middle=white, bottom=green
gt_image_normal = torch.ones((height, width, 3), device="cuda") * 1.0
gt_image_normal[: height // 3, :, :] = torch.tensor([1.0, 0.0, 0.0], device="cuda")     # top third: red
# middle third stays white
gt_image_normal[2 * height // 3 :, :, :] = torch.tensor([0.0, 1.0, 0.0], device="cuda")  # bottom third: green

# Reversed: top=green, middle=white, bottom=red
gt_image_reversed = torch.ones((height, width, 3), device="cuda") * 1.0
gt_image_reversed[: height // 3, :, :] = torch.tensor([0.0, 1.0, 0.0], device="cuda")     # top third: green
# middle third stays white
gt_image_reversed[2 * height // 3 :, :, :] = torch.tensor([1.0, 0.0, 0.0], device="cuda")  # bottom third: red

gt_image_normal_chw = gt_image_normal.permute(2, 0, 1).contiguous()      # [3, H, W]
gt_image_reversed_chw = gt_image_reversed.permute(2, 0, 1).contiguous()  # [3, H, W]

# ----------------------------------------------------------------------
# 5. Optimizer
# ----------------------------------------------------------------------
optimizer = torch.optim.Adam(
    [means3D, shs, opacity, scales, rotations],
    lr=0.01,
)

# ----------------------------------------------------------------------
# 6. Training loop (with periodic progress frames saved)
# ----------------------------------------------------------------------
os.makedirs("training_progress", exist_ok=True)

num_iters = 500
switch_point = num_iters // 2  # first half trains toward normal tricolor, second half toward reversed

for it in range(num_iters):
    # pick which ground truth to train toward this iteration
    current_gt_chw = gt_image_normal_chw if it < switch_point else gt_image_reversed_chw

    settings = GaussianRasterizationSettings(
        antialiasing=True,
        image_height=height,
        image_width=width,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),  # black background
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
        opacities=torch.sigmoid(opacity),                          # keep opacity in [0,1]
        scales=torch.abs(scales),                                  # keep scales positive
        rotations=torch.nn.functional.normalize(rotations, dim=-1),  # valid quaternion
        cov3D_precomp=None,
    )

    loss = torch.nn.functional.mse_loss(rendered_image, current_gt_chw)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it == 0:
        print("radii > 0:", (radii > 0).sum().item(), "/", num_gaussians)

    if it % 25 == 0 or it == num_iters - 1:
        with torch.no_grad():
            arr = rendered_image.detach().cpu().numpy()
            arr = np.transpose(arr, (1, 2, 0))
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(f"training_progress/iter_{it:04d}.png")

    if it % 50 == 0 or it == num_iters - 1:
        print(f"iter {it:4d}  loss {loss.item():.6f}")

torch.cuda.synchronize()

# ----------------------------------------------------------------------
# 7. Save final rendered image and ground truth for comparison
# ----------------------------------------------------------------------
arr = rendered_image.detach().cpu().numpy()      # [3, 600, 800]
arr = np.transpose(arr, (1, 2, 0))                # -> [600, 800, 3]
arr = (arr * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(arr).save("rendered_output.png")  # final state -> should resemble the REVERSED pattern

gt_normal_arr = (gt_image_normal.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(gt_normal_arr).save("gt_tricolor_normal.png")

gt_reversed_arr = (gt_image_reversed.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(gt_reversed_arr).save("gt_tricolor_reversed.png")

print("Saved rendered_output.png, gt_tricolor_normal.png, gt_tricolor_reversed.png")

# ----------------------------------------------------------------------
# 8. Stitch saved progress frames into an animated GIF
# ----------------------------------------------------------------------
frame_paths = sorted(glob.glob("training_progress/iter_*.png"))
frames = [Image.open(p) for p in frame_paths]

if frames:
    frames[0].save(
        "training_progress.gif",
        save_all=True,
        append_images=frames[1:],
        duration=150,   # ms per frame
        loop=0,         # loop forever
    )
    print(f"Saved training_progress.gif from {len(frames)} frames")
