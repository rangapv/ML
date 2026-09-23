#!/usr/bin/env bash
#author:rangapv@yahoo.com
#23-09-2026

installsh() {

input1="$#"
input2=("$@")
count1=0

for i in "${input2[@]}"
do
	((count1+=1))
	if (( $count1 > 4 ))
	then
	  #echo "executing install $i"
	  ins1=`sudo apt install -y $i`
	  echo "executed install for $i and the status is $?"
	  echo ""
	fi
done

}

installsh sudo apt-get -y install libglm-dev cmake libxcb-dri3-0 libxcb-present0      libpciaccess0 \
libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev g++ gcc \
libwayland-dev libxrandr-dev libxcb-randr0-dev libxcb-ewmh-dev \
git python-is-python3 bison libx11-xcb-dev liblz4-dev libzstd-dev \
ocaml-core ninja-build pkg-config libxml2-dev wayland-protocols python3-jsonschema \
clang-format qtbase5-dev qt6-base-dev qt6-wayland-dev
