#!/usr/bin/env bash
#author:rangapv@yahoo.com
#14-09-2026

#vlkver="1.3.296.0"
vlkver="1.4.357.0"

install_rt(){

vkins42=`cd ~/${vlkver}/;source ~/${vlkver}/setup-env.sh;vulkaninfo --summary`

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


step4() {

        vkstp41=`cp ./step4/01_foundation.cpp ./nvpro2/vk_raytracing_tutorial_KHR/raytrace_tutorial/01_foundation_copy/`
	vkstp42=`cp ./step4/shaderio.h ./nvpro2/vk_raytracing_tutorial_KHR/raytrace_tutorial/01_foundation_copy/shaders/`
        vkstp43=`cd nvpro2;cd vk_raytracing_tutorial_KHR;source ~/${vlkver}/setup-env.sh;cmake -B build -S .`
        vkstp44=`cd nvpro2;cd vk_raytracing_tutorial_KHR;source ~/${vlkver}/setup-env.sh;cmake --build build -j 8`
        #vkins45=`cd nvpro2/vk_raytracing_tutorial_KHR/_bin/Release;source ~/${vlkver}/setup-env.sh;./01_foundation_copy`


}

install_rt

vulkinfo

step0

step3

step4

