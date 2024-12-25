#!/usr/bin/env bash
#rangapv@yahoo.com
#20-10-24

#set -e

source <(curl -s https://raw.githubusercontent.com/rangapv/bash-source/main/s1.sh) > /dev/null 2>&1
#source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/refs/heads/main/libraries.sh) > /dev/null 2>&1
source <(curl -s https://raw.githubusercontent.com/rangapv/QuikFix/refs/heads/master/ver.sh) > /dev/null 2>&1


requirement() {

i1=`lspci | grep -i nvidia`
i1s="$?"

if [ "$i1s" == "1" ]
then
	echo "This system is not nvidia compatible"
	exit
else
	echo "This system is compatible ${i1}"
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

pipi_nstall() {

cl=("$@")
cln="$#"
pipv="pip3"
inscnt=0

echo "the total pip packages to check is $cln"

for i in "${cl[@]}"
do
wc=`$pipv show $i`
wcs="$?"

if [[ ( $wcs != "0" ) ]]
then
    echo "\"$i\"  is not installed , proceeding to install the same with $pipv" 
    ins_pippak=`$pipv install $i`
else
    echo "\"$i\" is installed proceeding with other checks"
fi
done

}

cuda_toolkit() {

#una=x86_64
#ki=ubuntu
#cmd1=apt-get

#irelease=`cat /etc/*-release | grep DISTRIB_RELEASE | awk '{split($0,a,"=");print a[2]}' |  awk '{split($0,a,".");print a[1]a[2]}'`

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
nv1=`/usr/local/cuda/bin/nvcc --version`
nv1s=$?
#echo "nv1 is $nv1 and nv1s is $nvi1"
if (( "$nv1s" == "0" ))
then
nvc1=`echo "$nv1" |grep release | awk '{split($0,a,","); print a[1]}'`
nvc2=`echo "$nv1" |grep release | awk '{split($0,a,","); print a[2]}'`
nvc21=`echo "$nvc2" |grep release | awk '{split($0,a," "); print a[2]}'`
nvc22=`echo "$nvc21" |awk '{split($0,a,"."); print a[1]}'`
nvc3=`echo "$nv1" |grep release | awk '{split($0,a,","); print a[3]}'`
#nvc1=`echo "$nv1" |grep release | awk '{split($0,a,","); print a[1]}'`
cuda_release=`echo "The $nvc1 is $nvc2"`
cuda_ver=`echo "The $nvc1 is version $nvc3"`
echo "$cuda_release"
echo "$cuda_ver"
echo "the cuda main relase is $nvc22"
#tensorRT-ver=``
else
	echo "looks-like nvidia-toolkit is not installed"
	exit
fi

}


cuda_cuDNN() {

#irelease=`cat /etc/*-release | grep DISTRIB_RELEASE | awk '{split($0,a,"=");print a[2]}' |  awk '{split($0,a,".");print a[1]a[2]}'`
#pre-requistise install zlib getting it from my repo ansible-install source above ...calling the install here

vercheck 1.3.1 "python3 -c \"import zlib;print(zlib.ZLIB_RUNTIME_VERSION)\"" zlibadd

#wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<arch>/cuda-keyring_1.1-1_all.deb
#sudo dpkg -i cuda-keyring_1.1-1_all.deb
cudkey1=`wget https://developer.download.nvidia.com/compute/cuda/repos/${ki}${irelease}/${una}/cuda-keyring_1.1-1_all.deb`
cudkey2=`sudo dpkg -i cuda-keyring_1.1-1_all.deb`
update1=`sudo apt-get update`
#update2=`sudo cp /var/cudnn-local-*/cudnn-*-keyring.gpg /usr/share/keyrings/`

#sudo apt-get -y install cudnn9-cuda-12
#if you have installed the CUDA tool-kit form this program then you would have had the key-ring & other packages installed
#cudnnv="12"
#get the cuda-tookkit Installed version
#
#
#
nvidia_version
cuDNN=`sudo $cmd1 install cudnn9-cuda-${nvc22}`
#cuDNNsamp=`sudo $cmd1 install libcudnn9-samples /usr/local/src`
#cuDNNdev=`sudo $cmd1 install libcudnn9-dev /usr/local/src`
cudnn_version="9.6.0"
cuda_version="cuda${nvc21}"

#TO-PIN it to a particular verison uncomment the following
#sudo $cmd1 install libcudnn9=${cudnn_version}-1+${cuda_version}
#sudo $cmd1 install libcudnn9-dev=${cudnn_version}-1+${cuda_version}
#sudo $cmd1 install libcudnn9-samples=${cudnn_version}-1+${cuda_version}

}


firstalternative_cuDNN(){

distro="${ki}${irelease}"
architecture="amd64"
cdnv="9.6.0"
f1c=`wget https://developer.download.nvidia.com/compute/cudnn/$cdnv/local_installers/cudnn-local-repo-$distro-${cdnv}_1.0-1_$architecture.deb`
f2c=`sudo dpkg -i cudnn-local-repo-$distro-${cdnv}_1.0-1_$architecture.deb`
f3c=`sudo cp /var/cudnn-local-*/cudnn-*-keyring.gpg /usr/share/keyrings/`
f4c=`sudo apt-get update`
f5c=`sudo apt-get -y install cudnn9-cuda-12`

}


verify_cuDNN() {

vrcdnn1=`cp -r /usr/src/cudnn_samples_v9/ $HOME`
vrcdnn2=`cd $HOME/cudnn_samples_v9/mnistCUDNN;make clean && make`
vrcdnn3=`cd $HOME/cudnn_samples_v9/mnistCUDNN;./mnistCUDNN`
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

#irelease=`cat /etc/*-release | grep DISTRIB_RELEASE | awk '{split($0,a,"=");print a[2]}' |  awk '{split($0,a,".");print a[1]a[2]}'`

version="10.x.x.x"
arch=$(uname -m)
#version="10.6.0.26"
cuda="cuda-12.2"

tensort_ins1="TensorRT-10.6.0.26.Linux.x86_64-gnu.cuda-11.8.tar.gz"
version=`echo "${tensort_ins1}" | awk '{split($0,a,"-");print a[2]}'| grep -Eo '^[0-9.]*' | sed 's/.$//'`

tensort_ins2=`echo "${tensort_ins1}" | grep -o "cuda-.*"`
tensort_ins3=${tensort_ins2:0:-7}
cuda="${tensort_ins3}"

#tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz

step1=`tar -xzvf TensorRT-${version}.Linux.${una}-gnu.${cuda}.tar.gz`

step2=`export LD_LIBRARY_PATH=<TensorRT-${version}/lib>:$LD_LIBRARY_PATH`

step3=`cd TensorRT-${version}/python`

step4=`python3 -m pip3 install tensorrt-*-cp3x-none-linux_x86_64.whl`

step51=`python3 -m pip3 install tensorrt_lean-*-cp3x-none-linux_x86_64.whl`

step52=`python3 -m pip3 install tensorrt_dispatch-*-cp3x-none-linux_x86_64.whl`

}

verify_tensorRT() {

verify1=`dpkg-query -W tensorrt`
verify1s="$?"

if [ $verify1s == "0" ]
then
	echo "TensorRT install successful"
	echo "The installed version is $verify1"
else
	echo "The install of tensorRT failed"
	echo "$verify1"
fi

}

first-alternate-installtensorRT() {

#trti1=`wget https://developer.download.nvidia.com/compute/cuda/repos/${ki}${irelease}/${una}/cuda-keyring_1.1-1_all.deb`
#trti2=`sudo dpkg -i cuda-keyring_1.1-1_all.deb`

tag="${tensorRTv}-${cudaVer}"

tnsrt1=`sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb`
tnsrt2=`sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/`
tnsrt3=`sudo $cmd1 update`

tnsrt4=`sudo $cmd1 install tensorrt`

}


onnx_install() {

whonnx="$@"

if [ -z  "$@" ]
then
     onx_cpu=`pipi_nstall onnxruntime`
elif [ "$@" == "gpu" ]
then
     onx_gpu1=`pipi_nstall onnxruntime-gpu`

elif [ "$@" == "qnn" ]
then
     onx_qnn=`pipi_nstall onnxruntime-qnn`
else
	echo "No-compatible option to install ONNX runtime"
fi

torch_onnx=`python3 -c "import torch;print (torch.__version__)"`
tor_onxs="$?"

if [ "$tor_onxs" != "0" ]
then
  instor_onnx=`pipi_nstall torch`
fi

tflow_onnx=`python3 -c "import tf2onnx;print (tf2onnx.__version__)"`
tflow_onxs="$?"

if [ "$tflow_onxs" != "0" ]
then
  instflo_onnx=`pipi_nstall tf2onnx`
fi

skl2onnx_onnx=`python3 -c "import skl2onnx;print (skl2onnx.__version__)"`
skl2onnx_onxs="$?"

if [ "$skl2onnx_onxs" != "0" ]
then
  insskl_onnx=`pipi_nstall skl2onnx`
fi

}



#checking to see if python and pip are installed before installing cuda for python

install(){

chkifinsta python3 pip3 gcc

cuda_toolkit

cuda_cuDNN

verify_cuDNN

verify_tensorRT

pipi_nstall onnx

onnx_install gpu

}

#Testing functions...
#requirement

#chkifinsta python3 pip3 gcc

#cuda_toolkit

#nvidia_version
#
#cuda_python
#cuda_cuDNN
#firstalternative_cuDNN
verify_cuDNN
