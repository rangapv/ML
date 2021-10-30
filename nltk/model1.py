#!/usr/bin/env python
import os
import subprocess

# clone the repository
if not os.path.exists('MODNet'):
   print("Hello")
   pc = "git clone https://github.com/ZHKKKe/MODNet"
   pl = subprocess.run(pc, capture_output=True, shell=True, text=True, check=False)
   l21 = pl.stdout
   print(l21)


# dowload the pre-trained ckpt for image matting
pretrained_ckpt = './MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
if not os.path.exists(pretrained_ckpt):
   gd = "gdown --id 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz -O pretrained/modnet_photographic_portrait_matting.ckpt" 
   ps = subprocess.run(gd, capture_output=True, shell=True, text=True, check=False)
   l22 = ps.stdout
   print(l22)
