#!/usr/bin/env bash
#author:rangapv@yahoo.com
#23-10-2026

vert1=`~/1.4.357.0/setup-env.sh;~/1.4.357.0/x86_64/bin/glslc shader.vert -o vert.spv`
vert1s="$?"
frag1=`~/1.4.357.0/setup-env.sh;~/1.4.357.0/x86_64/bin/glslc shader.frag -o frag.spv`
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
