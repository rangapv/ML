#!/usr/bin/env bash
#rangapv@yahoo.com
#20-10-24

#set -e

source <(curl -s https://raw.githubusercontent.com/rangapv/bash-source/main/s1.sh) > /dev/null 2>&1
source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/refs/heads/main/libraries.sh) > /dev/null 2>&1
source <(curl -s https://raw.githubusercontent.com/rangapv/QuikFix/refs/heads/master/ver.sh) > /dev/null 2>&1


requirement() {

i1=`lspci | grep -i nvidia`
i1s="$?"

if [ "$i1s" == "1" ]
then
	echo "This system is not nvidia compatible"
	exit
fi

}


chkifinsta() {

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
    echo "\"$i\" is installed proceeding with other checks"
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

cuda_toolkit() {

#una=x86_64
#ki=ubuntu
#cmd1=apt-get

irelease=`cat /etc/*-release | grep DISTRIB_RELEASE | awk '{split($0,a,"=");print a[2]}' |  awk '{split($0,a,".");print a[1]a[2]}'`

ctk1=`wget https://developer.download.nvidia.com/compute/cuda/repos/${ki}${irelease}/${una}/cuda-${ki}${irelease}.pin`
ctk2=`sudo mv cuda-${ki}${irelease}.pin /etc/apt/preferences.d/cuda-repository-pin-600`
ctk3=`wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-${ki}${irelease}-12-6-local_12.6.2-560.35.03-1_amd64.deb`
ctk4=`sudo dpkg -i cuda-repo-${ki}${irelease}-12-6-local_12.6.2-560.35.03-1_amd64.deb`
ctk5=`sudo cp /var/cuda-repo-${ki}${irelease}-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/`
ctk6=`sudo $cmd1 update`
ctk7=`sudo $cmd1 -y install cuda-toolkit-12-6`

}

cuda_python(){

echo "hello"

lnxdr=`uname -v | awk '{split($0,a," "); print a[1]}'`

pypip="pip3"

#install
cupy1=`${pypip} install cuda-python`
cupy1s="$?"

}

nvidia_version() {

nv1=`nvcc --version`

cuda_ver=``
tensorRT-ver=``

}


cuda_cuDNN() {

#pre-requistise install zlib getting it from my repo ansible-install source above ...calling the install here

vercheck 1.3.1 "python3 -c \"import zlib;print(zlib.ZLIB_RUNTIME_VERSION)\"" zlibadd

#if you have installed the CUDA tool-kit form this program then you would have had the key-ring & other packages installed
#cudnnv="12"
#cuDNN=`sudo $cmd1 -y install cudnn-cuda-${cudnnv}`

cudnn_version="8.9.7"
cuda_version="cuda12.2"

sudo $cmd1 install libcudnn8=${cudnn_version}-1+${cuda_version}
sudo $cmd1 install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
sudo $cmd1 install libcudnn8-samples=${cudnn_version}-1+${cuda_version}

}

verify_cuDNN() {

vrcdnn1=`cp -r /usr/src/cudnn_samples_v8/ $HOME`
vrcdnn2=`cd $HOME/cudnn_samples_v8/mnistCUDNN;make clean && make`
vrcdnn3=`cd $HOME/cudnn_samples_v8/mnistCUDNN;./mnistCUDNN`
echo "the test result is $vrcdnn3"

}

tensorrt_install() {

# Download the tensorRT https://developer.nvidia.com/tensorrt
# https://developer.nvidia.com/tensorrt/download/10x

#una=x86_64
#ki=ubuntu
#cmd1=apt-get

#os="ubuntuxx04"
os="${ki}${irelease}"
#tag="10.x.x-cuda-x.x"


irelease=`cat /etc/*-release | grep DISTRIB_RELEASE | awk '{split($0,a,"=");print a[2]}' |  awk '{split($0,a,".");print a[1]a[2]}'`

version="10.x.x.x"
arch=$(uname -m)
version="10.6.0.26"
cuda="cuda-12.2"

#tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz

step1=`tar -xzvf TensorRT-${version}.Linux.${una}-gnu.${cuda}.tar.gz`

step2=`export LD_LIBRARY_PATH=<TensorRT-${version}/lib>:$LD_LIBRARY_PATH`

step3=`cd TensorRT-${version}/python`

step4=`python3 -m pip3 install tensorrt-*-cp3x-none-linux_x86_64.whl`

step51=`python3 -m pip3 install tensorrt_lean-*-cp3x-none-linux_x86_64.whl`

step52=`python3 -m pip3 install tensorrt_dispatch-*-cp3x-none-linux_x86_64.whl`

}

first-alternate-installtensorRT() {

#trti1=`wget https://developer.download.nvidia.com/compute/cuda/repos/${ki}${irelease}/${una}/cuda-keyring_1.1-1_all.deb`
#trti2=`sudo dpkg -i cuda-keyring_1.1-1_all.deb`

tag="${tensorRTv}-${cudaVer}"

tnsrt1=`sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb`
tnsrt2=`sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/`
tnsrt3=`sudo $cmd1 update`

tnsrt4=`sudo $cmd1 install tensorrt`

verify1=`dpkg-query -W tensorrt`
verify1s="$?"

if [ $verify1s == "0" ]
then
	echo "TensorRT install successful"
	echo "The installed version is $verify1"
else
	echo "The install of tensorRT failed"
	echo "$verofy1"
fi

}

second-alternate-installtensorRT() {

tag="${tensorRTv}-${cudaVer}"

tnsrt1=`sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb`
tnsrt2=`sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/`
tnsrt3=`sudo $cmd1 update`

tnsrt4=`sudo $cmd1 install tensorrt`

verify1=`dpkg-query -W tensorrt`
verify1s="$?"

if [ $verify1s == "0" ]
then
        echo "TensorRT install successful"
        echo "The installed version is $verify1"
else
        echo "The install of tensorRT failed"
        echo "$verofy1"
fi

}
#checking to see if python and pip are installed before installing cuda for python

install(){

chkifinsta python3 pip3 gcc

cuda_toolkit

cuda_cuDNN

verify_cuDNN

}

