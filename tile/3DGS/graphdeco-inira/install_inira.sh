#!/usr/bin/env bash
#author:rangapv@yahoo.com
#24-08-2026

install_inira(){

gi1=`pip3 uninstall torch torchvision torchaudio`
gi2=`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
gi3=`pip3 install plyfile opencv-python`

si1=`sudo apt install gcc-11 g++-11`
#si2=`sudo ln -sf /usr/bin/gcc-11 /usr/bin/gcc`
si123=`sudo apt install unzip`

si3=`export NVCC_FLAGS="-allow-unsupported-compiler"`

gi4=`git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive`

gi5=`cd gaussian-splatting;pip3 install ./submodules/diff-gaussian-rasterization --no-build-isolation`

gi6=`cd gaussian-splatting;pip3 install ./submodules/simple-knn --no-build-isolation`

si4=`cd gaussian-splatting;wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip`

si5=`cd gaussian-splatting;unzip tandt_db.zip`

si5=`cd gaussian-splatting;python3 train.py -s ./tandt/train`

}

build_check(){

cmd1=("$@")
tcmd="$#"
insdep=0

echo "the total depedency to check is $tcmd"

for i in "${cmd1[@]}"
do

wc=`which $i`
wcs="$?"

if [[ ( $wcs == "0" ) ]]
then
	v1=`($i -V | awk -v pk1="$i" '{split($2,a,"."); if (a[1] == "3") { if (a[2] == "10") print pk1 (" verison satisfies gapghdeco-inira Install") } }')`
    echo "$v1"
    echo "\"$i\" is installed @ $wc and its version is `$i -V` proceeding with other checks"
else
    echo "\"$i\"  is not installed .pls install it and then re-run this script for other tasks"
    insdep=1
fi

done

if (( $insdep == 1 ))
then
   echo "Install all the dependencies and proceed after, exiting now"
   exit
else
   echo "All the dependecy \" ${cmd1[@]} \" are installed"
fi

}

check_gcc (){
i1="gcc"
i11=`which $i1`
i11s="$?"
if [[ ( $i11s == "0" ) ]]
then
#`gcc --version | grep gcc |awk '{split($0,a," "); print a[NF]}'`
v2=`gcc --version | grep gcc | awk -v pk2="$i1" '{split($3,a,"."); if (a[1] == "11") { print pk2 (" version satisifies graphdeco-inira install") }}'`
echo "$v2"
else
	echo "$i1 is not the required version for graphdeco-inira"
fi
}


build_check python3
check_gcc
install_inira
