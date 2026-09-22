#!/usr/bin/env bash
#author:rangapv@yahoo.com
#14-09-2026

install_rt(){

vkins1=`sudo apt-get -y install libtbb-dev`
vkins11=`sudo apt install -y x11-apps libx11-dev xserver-xorg-dev xorg-dev`
vkins12=`sudo apt install -y xz-utils`
vkins13=`sudo apt install -y libxinerama-dev libxi-dev`
vkins14=`sudo apt install -y libxcursor-dev` 
vkins2=`sudo apt install -y libxcb-xinput0 libxcb-xinerama0 libxcb-cursor-dev`
vkins3=`sudo apt-get -y install libglm-dev cmake libxcb-dri3-0 libxcb-present0      libpciaccess0 \
libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev g++ gcc \
libwayland-dev libxrandr-dev libxcb-randr0-dev libxcb-ewmh-dev \
git python-is-python3 bison libx11-xcb-dev liblz4-dev libzstd-dev \
ocaml-core ninja-build pkg-config libxml2-dev wayland-protocols python3-jsonschema \
clang-format qtbase5-dev qt6-base-dev qt6-wayland-dev`

vkins4=`wget https://sdk.lunarg.com/sdk/download/1.4.357.1/linux/vulkansdk-linux-x86_64-1.4.357.1.tar.xz`

vkins5=`wget https://sdk.lunarg.com/sdk/download/1.4.357.1/linux/config.json`

vkins41=`tar -xvf ./vulkansdk-linux-x86_64-1.4.357.1.tar.xz -C ~/`

vkins42=`cd ~/1.4.357.1/;source ~/1.4.357.1/setup-env.sh;vulkaninfo`

vkins6=`mkdir nvpro2;cd nvpro2; git init; git clone https://github.com/nvpro-samples/nvpro_core2.git`

vkins7=`cd nvpro2;git clone https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR.git`

vkins8=`cd nvpro2;cd vk_raytracing_tutorial_KHR;source ~/1.4.357.1/setup-env.sh;cmake -B build -S .`

vkins9=`cd nvpro2;cd vk_raytracing_tutorial_KHR;source ~/1.4.357.1/setup-env.sh;cmake --build build -j 8`

vkins10=`cd nvpro2;cd vk_raytracing_tutorial_KHR;source ~/1.4.357.1/setup-env.sh;./_bin/01_Foundation`

}

install_rt
