#!/usr/bin/env python3
#author:rangapv@yahoo.com
#30-08-2026

#A simple program to test the initial setup of the graphdeco-inira pacakges and the working in the CUDA environment
#output: It prints the rasterized Matrix output and the Shape of the Tensor holding the final output image and 
#also the Orignal Imgae construct as well as the Rasterized Image of the output

import torch
import math
import glob
from torch import Tensor, optim
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from PIL import Image
import numpy as np

class Diffgaus:
# 1. Set up basic image dimensions

   def __init__(self, gt_image: Tensor, gt_image_reversed_chw: Tensor):
    self.device = torch.device("cuda:0")
    self.gt_image = gt_image.to(device=self.device)
    self.gt_image_reversed_chw= gt_image_reversed_chw.to(device=self.device)
    #print(f'the shape of target is {gt_image1.shape}')
    self.width = 400
    self.height = 300
    self.num_gaussians = 2000
# 2. Create dummy 3D Gaussian properties (Move to GPU)
    self.means3D = torch.zeros((100, 3), device="cuda")
    self.shs = torch.zeros((100, 16, 3), device="cuda")  # Spherical Harmonics colors
    self.opacity = torch.ones((100, 1), device="cuda")
    self.scales = torch.ones((100, 3), device="cuda")
    self.rotations = torch.zeros((100, 4), device="cuda")  # Quaternions
    self.rotations[:, 0] = 1.0

    self.means3D.requires_grad_(True)
    self.shs.requires_grad_(True)
    self.opacity.requires_grad_(True)
    self.scales.requires_grad_(True)
    self.rotations.requires_grad_(True)

    fovx = 2 * math.atan(0.5)
    fovy = 2 * math.atan(0.5)

    self.viewmatrix = torch.eye(4, device="cuda")
    self.viewmatrix[2, 3] = 3.0  # move camera back
    self.viewmatrix = self.viewmatrix.transpose(0, 1).contiguous()

    self.projmatrix = self._getProjectionMatrix(0.01, 100.0, fovx, fovy).cuda()
    self.projmatrix = self.projmatrix.transpose(0, 1).contiguous()
    self.projmatrix = self.viewmatrix.unsqueeze(0).bmm(self.projmatrix.unsqueeze(0)).squeeze(0)

    self.cam_pos = torch.tensor([0.0, 0.0, -5.0], device="cuda")

    self._tensorinit()

# 3. Set up camera matrices (Dummy 4x4 identities for example)
   def _getProjectionMatrix(self,znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    P = torch.zeros(4, 4)
    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

   def _tensorinit(self):
       
    num_iters = 500
#    switch_point = num_iters // 2  # first half trains toward normal tricolor, second half toward reversed
    self.gt_image1 = self.gt_image.permute(2, 0, 1).contiguous()
    optimizer = torch.optim.Adam(
     [self.means3D, self.shs, self.opacity, self.scales, self.rotations],
     lr=0.01,
    )

    switch_point = num_iters // 2  # first half trains toward normal tricolor, second half toward reversed

    # 5. Initialize rasterizer and render
    self.settings = GaussianRasterizationSettings(
       antialiasing=True,
       image_height=self.height,
       image_width=self.width,
       tanfovx=0.5,
       tanfovy=0.5,
       bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), # Black background
       scale_modifier=1.0,
       viewmatrix=self.viewmatrix,
       projmatrix=self.projmatrix,
       sh_degree=3,
       campos=self.cam_pos,
       prefiltered=False,
       debug=False
    )   
    rasterizer = GaussianRasterizer(raster_settings=self.settings)
    for it in range(num_iters):
      current_gt_chw = self.gt_image if it < switch_point else self.gt_image_reversed_chw
      # pick which ground truth to train toward this iteration

      #self.rasterizer = GaussianRasterizer(raster_settings=self.settings)
      self.rendered_image, self.radii, _ = rasterizer(
       means3D=self.means3D,
       means2D=torch.zeros_like(self.means3D, requires_grad=True),
      # means2D=torch.zeros((100, 2), device="cuda", requires_grad=True), # Keeps track of gradients
       shs=self.shs,
       colors_precomp=None,
       opacities=self.opacity,
       scales=self.scales,
       rotations=self.rotations,
       cov3D_precomp=None
      ) 

      #loss = torch.nn.functional.mse_loss(self.rendered_image,self.gt_image1)
      loss = torch.nn.functional.mse_loss(self.rendered_image, current_gt_chw)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if it == 0:
        print("radii > 0:", (self.radii > 0).sum().item(), "/", self.num_gaussians)

      if it % 25 == 0 or it == num_iters - 1:
        with torch.no_grad():
            arr = self.rendered_image.detach().cpu().numpy()
            arr = np.transpose(arr, (1, 2, 0))
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(f"training_progress/iter_{it:04d}.png")

      if it % 50 == 0 or it == num_iters - 1:
        print(f"iter {it:4d}  loss {loss.item():.6f}")

    torch.cuda.synchronize()
# 'rendered_image' is a PyTorch tensor ready for backpropagation!
    print(self.rendered_image) # Output layout: [3, height, width]
#print(rendered_image.shape) # Output layout: [3, height, width]
    print(self.rendered_image.shape) # Output layout: [3, height, width]
    self._array2img()

   def _array2img(self):
     arr = self.rendered_image.detach().cpu().numpy()      # [3, 600, 800]
     arr = np.transpose(arr, (1, 2, 0))                # -> [600, 800, 3]
     arr = (arr * 255).clip(0, 255).astype(np.uint8)
     Image.fromarray(arr).save("rendered_output.png")  # final state -> should resemble the REVERSED pattern

     gt_normal_arr = (self.gt_image.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
     Image.fromarray(gt_normal_arr).save("gt_tricolor_normal.png")

     gt_reversed_arr = (self.gt_image_reversed_chw.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
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
        duration=450,   # ms per frame
        loop=0,         # loop forever
       ) 
     print(f"Saved training_progress.gif from {len(frames)} frames")

def main():
    height = 300 
    width = 400 
    gt_image = torch.ones((height,width,3)) * 1.0
    # make top left and bottom right red, blue
    # top third: red
    gt_image[: height // 3, :, :] = torch.tensor([1.0, 0.0, 0.0])
    # middle third stays white (no need to set it, already white)
    gt_image[2 * height // 3 :, :, :] = torch.tensor([0.0, 1.0, 0.0], device="cuda")  
    # bottom third: green
    gt_image[2 * height // 3 :, :, :] = torch.tensor([0.0, 1.0, 0.0])

    gt_image_reversed = torch.ones((height, width, 3), device="cuda") * 1.0
    gt_image_reversed[2 * height // 3 :, :, :] = torch.tensor([1.0, 0.0, 0.0], device="cuda")  # bottom third: red
    gt_image_reversed[: height // 3, :, :] = torch.tensor([0.0, 1.0, 0.0], device="cuda")     # top third: green
    # middle third stays white
    gt_image = gt_image.permute(2, 0, 1).contiguous()      # [3, H, W]
    gt_image_reversed_chw = gt_image_reversed.permute(2, 0, 1).contiguous()  # [3, H, W]

    print(f"the gt size is {gt_image.size}")
    tr = Diffgaus(gt_image,gt_image_reversed_chw)

if __name__=="__main__":
    main()

