#!/usr/bin/env python3
#author:rangapv@yahoo.com
#20-02-2026

#draw tri-color with Strategy
#works for both 2d and 3d
#usuage: ./filename --model_type=2dgs/3dgs

import math
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs
from gsplat import DefaultStrategy

class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        #colors = torch.rand((100, 3), device=device)
        self.colors = torch.rand(self.num_points, 3, device=self.device)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        d = 3
        #self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.colors.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        #self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False
        
        self.params = torch.nn.ParameterDict({"means": self.means, "scales": self.scales, "opacities": self.opacities, "quats": self.quats,"colors": self.colors})
        lr: float = 0.01
        self.optimizer = {k: torch.optim.Adam([p], lr=lr) for k, p in self.params.items()}

        self.strategy = DefaultStrategy()
        # Check the sanity of the parameters and optimizers
        self.strategy.check_sanity(self.params, self.optimizer)

        # Initialize the strategy state
        self.strategy_state = self.strategy.initialize_state()

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        model_type: Literal["3dgs", "2dgs"] = "3dgs",
    ):
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 2  # rasterization, backward
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )

        if model_type == "3dgs":
            rasterize_fnc = rasterization
            print(f'This is 3dgs Rasterization') 
        elif model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs
            print(f'This is 2dgs Rasterization') 

        for iter in range(iterations):
            start = time.time()
            renders = rasterize_fnc(
                means=self.means,
                quats=self.quats / self.quats.norm(dim=-1, keepdim=True),
                scales=self.scales,
                opacities=torch.sigmoid(self.opacities),
                colors=torch.sigmoid(self.colors)[None],
                viewmats=self.viewmat[None],
                Ks=K[None],
                width=self.W,
                height=self.H,
                packed=False,
            )
            info = renders[-1]
            if iter == -1:
               print(f'The output of rasterization is {renders}, its length is {len(renders)}')
               print(f'the output of renders[0] is {renders[0]}')
               print(f'the output of renders[1] is {renders[1]}')
               print(f'the output of renders[2] is {renders[2]}')
               print(f'the output of renders[0][0] is {renders[0][0]}')
            self.strategy.step_pre_backward(self.params, self.optimizer, self.strategy_state, iter, info)

            out_img = renders[0][0]
            torch.cuda.synchronize()
            times[0] += time.time() - start
            loss = mse_loss(out_img, self.gt_image)
            for opt in self.optimizer.values():
                opt.zero_grad()
            start = time.time()
            loss.backward()
 
            self.strategy.step_post_backward(self.params, self.optimizer, self.strategy_state, iter, info)

            torch.cuda.synchronize()
            times[1] += time.time() - start
            for opt in self.optimizer.values():
                opt.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
        )

def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue

        # top third: red
        gt_image[: height // 3, :, :] = torch.tensor([1.0, 0.0, 0.0])

        # middle third stays white (no need to set it, already white)

        # bottom third: green
        gt_image[2 * height // 3 :, :, :] = torch.tensor([0.0, 1.0, 0.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
        model_type=model_type,
    )

if __name__ == "__main__":
    tyro.cli(main)
