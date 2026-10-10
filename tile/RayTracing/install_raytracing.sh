#!/usr/bin/env bash
#author:rangapv@yahoo.com
#14-09-2026

#vlkver="1.3.296.0"
vlkver="1.4.357.0"

vulk_depend() {

echo "installing dependencies"
vkins01=`sudo apt-get update`
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

}

install_rt(){

vkins4=`wget https://sdk.lunarg.com/sdk/download/${vlkver}/linux/vulkansdk-linux-x86_64-${vlkver}.tar.xz`

vkins5=`wget https://sdk.lunarg.com/sdk/download/${vlkver}/linux/config.json`

vkins41=`tar -xvf ./vulkansdk-linux-x86_64-${vlkver}.tar.xz -C ~/`

vkins42=`cd ~/${vlkver}/;source ~/${vlkver}/setup-env.sh;vulkaninfo`

vkins6=`mkdir nvpro2;cd nvpro2; git init; git clone https://github.com/nvpro-samples/nvpro_core2.git`

vkins7=`cd nvpro2;git clone https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR.git`

#vkins8=`cd nvpro2;cd vk_raytracing_tutorial_KHR;source ~/${vlkver}/setup-env.sh;cmake -B build -S .`

#vkins9=`cd nvpro2;cd vk_raytracing_tutorial_KHR;source ~/${vlkver}/setup-env.sh;cmake --build build -j 8`

#vkins10=`cd nvpro2/vk_raytracing_tutorial_KHR/_bin/Release;source ~/${vlkver}/setup-env.sh;./01_foundation`

}

vulkinfo () {

vlkinfo=`vulkaninfo --summary`
echo "$vlkinfo"

}

step0() {

        vkstp1=`cd nvpro2/vk_raytracing_tutorial_KHR/raytrace_tutorial ; cp -r 01_foundation 01_foundation_copy`
       # vkstp2=`
	file2="nvpro2/vk_raytracing_tutorial_KHR/CMakeLists.txt"
        line21="add_subdirectory(raytrace_tutorial/01_foundation)"
        line22="add_subdirectory(raytrace_tutorial/01_foundation);add_subdirectory(raytrace_tutorial/01_foundation_copy)"
	repl12=`sudo sed -i "/^add_subdirectory(raytrace_tutorial\/01_foundation)$/a add_subdirectory(raytrace_tutorial/01_foundation_copy)" "$file2"`
}

step3() {
	vkstp31=`cp ./step3/01_foundation.cpp ./nvpro2/vk_raytracing_tutorial_KHR/raytrace_tutorial/01_foundation_copy/`
        vkstp4=`cd nvpro2;cd vk_raytracing_tutorial_KHR;source ~/${vlkver}/setup-env.sh;cmake -B build -S .`
        vkstp5=`cd nvpro2;cd vk_raytracing_tutorial_KHR;source ~/${vlkver}/setup-env.sh;cmake --build build -j 8`
	#vkins10=`cd nvpro2/vk_raytracing_tutorial_KHR/_bin/Release;source ~/${vlkver}/setup-env.sh;./01_foundation_copy`

}

startover() {

vrm1=`rm -r ~/1.4.357.0`
vrm2=`rm -r ~/ml1/tile/RayTracing/nvpro2`
vrm21=`rm -r ~/ml1/tile/RayTracing/config*`
vrm3=`rm -r ~/ml1/tile/RayTracing/vulkansdk-linux-x86_64-1.4.357.0.tar*`

}

vulk_depend

install_rt

vulkinfo

step0

step3

#startover
