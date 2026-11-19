#!/usr/bin/env bash
#author:rangapv@yahoo.com
#06-11-2026

insta_dep() {

dep1=`sudo apt update` 
dep2=`sudo apt install -y build-essential graphviz cmake git pkg-config`

}

insta_ggml() {
insggm1=`git clone https://github.com/ggml-org/ggml`
insggm2=`cd ggml;mkdir build && cd build;cmake ..;cmake --build . --config Release -j 8`
}

insta_ggml_all() {
echo "This requires Vulkan glsc shaders to compile..make sure you have Vulkan installed"
vulver="1.4.357.0"
insggmla1=`sudo apt update`
insggmla2=`sudo apt install -y build-essential cmake git libopenblas-dev pkg-config`
insggmla3=`git clone https://github.com/ggml-org/ggml`
insggmla4=`source ~/${vulver}/setup-env.sh;cd ggml;mkdir build;cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_VULKAN=ON \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_BUILD_EXAMPLES=ON`
insggmla5=`cmake --build ./ggml/build --config Release -j2`
}

complie-sample() {
samp1=`cd /ggml/build ;cmake --build . --target simple-backend -j2`
}

insta_dep

#Uncomment just one of the type of ggml installs
insta_ggml

#insta_ggml_all

compile-sample

