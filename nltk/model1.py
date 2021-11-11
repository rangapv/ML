#!/usr/bin/env python
import os
import subprocess
import shutil

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

# clean and rebuild the image folders
input_folder = './input'
if os.path.exists(input_folder):
   print("File with images for background extraction exists")
   # shutil.rmtree(input_folder)
else:
   print("Folder with images DOES NOT Exist") 
   #os.makedirs(input_folder)
output_folder = './output'
if os.path.exists(output_folder):
  shutil.rmtree(output_folder)
os.makedirs(output_folder)
# upload images (PNG or JPG)
#image_names = list(files.upload().keys())
#for image_name in image_names:
#  shutil.move(image_name, os.path.join(input_folder, image_name))


#app = 'python -m ./MODNet/demo/image_matting/colab/inference --input-path ,/input --output-path ./output --ckpt-path ./pretrained/modnet_photographic_portrait_matting.ckpt'
#pt = subprocess.run(app, capture_output=True, shell=True, text=True, check=False)
#l23 = pt.stdout
#print(pt)
#print(l23)

