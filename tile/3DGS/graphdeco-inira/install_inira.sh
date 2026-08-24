#!/usr/bin/env bash
#author:rangapv@yahoo.com
#24-08-2026

install_inira(){


gi1=`pip3 uninstall torch torchvision torchaudio`
gi2=`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
gi3=`pip3 install plyfile opencv-python`

si1=`sudo apt install gcc-11 g++-11`
si2=`sudo ln -sf /usr/bin/gcc ./gcc-11`
si123=`sudo apt install unzip`

si3=`export NVCC_FLAGS="-allow-unsupported-compiler"`


gi4=`git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive`

gi5=`cd gaussian-splatting;pip3 install ./submodules/diff-gaussian-rasterization --no-build-isolation`

gi6=`cd gaussian-splatting;pip3 install ./submodules/simple-knn --no-build-isolation`


si4=`cd gaussian-splatting;wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip`

si5=`cd gaussian-splatting;unzip tandt_db.zip`

si5=`cd gaussian-splatting;python3 train.py -s ./tandt/train`


}


install_inira
