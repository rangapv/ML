#!/usr/bin/env bash
#author:rangapv@yahoo.com
#23-10-2026
#READ-ME: This is for MacOS

VK_SDK_HOME="/Users/rangaswamypv/VulkanSDK/1.4.363.0"

vert1=`${VK_SDK_HOME}/setup-env.sh;${VK_SDK_HOME}/macos/bin/glslc shader.vert -o vert.spv`
vert1s="$?"
frag1=`${VK_SDK_HOME}/setup-env.sh;${VK_SDK_HOME}/macos/bin/glslc shader.frag -o frag.spv`
frag1s="$?"

if [ $vert1s == "0" ]
then
	echo "Vertex shaders has been compiled"
else
	echo "error in Vertex shader compilation"
fi

if [ $frag1s == "0" ]
then
	echo "Fragment shaders has been compiled"
else
	echo "Error in Fragment shader compilation"
fi
