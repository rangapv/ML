#!/usr/bin/env bash
#rangapv@yahoo.com
#20-10-24

#set -e


source <(curl -s https://raw.githubusercontent.com/rangapv/bash-source/main/s1.sh) > /dev/null 2>&1
source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/refs/heads/main/libraries.sh) > /dev/null 2>&1

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

}

nvidia_version() {

nv1=`nvcc --version`

cuda_ver=``
tensorRT-ver=``

}

versioncheck() {

     pkg="$@"	
     pyv=`which $pkg`
     pyvs="$?"
     if [[ ( $pyvs -ne 0 ) ]]
     then
            echo "No $pkg Found; GREENFIELD Installs proceeding"
     else
        vercheck $pkg 
        if [[ ( $piver1 = $cmd1 ) ]]
        then
                echo "Requirement satisfied Python is already in version \"${piver1}\" "
                exit
        elif [[ ( $piver1 < $cmd1 ) ]]
        then
                echo "Upgrading Python to $cmd1"

        elif [[ ( $piver1 > $cmd1 ) ]]
        then
                echo "The current version of Python ${piver1} is Higher than the request $cmd1 ;exiting"
                exit
        fi
     fi
}



vercheck() {

vrchk1="$@"
vrchk2=`${vrchk1} -c "import zlib ; print(zlib.ZLIB_RUNTIME_VERSION)"`
vrarg1=`echo ${vrchk2} | awk '{split($0,a,"."); print a}'`
newver="1.3.1"

vrarg3=($(echo ${vrchk2} | awk '{len=split($0,a,"."); for (n=0;n<=len;n++) print a[n]}'))
vrang4=($(echo ${newver} | awk '{len=split($0,a,"."); for (n=0;n<=len;n++) print a[n]}'))
lvrarg3=${#vrarg3[@]}
lvrarg4=${#vrarg4[@]}

if [ $newver == $vrchk2 ]
then
	echo "No upgrade required"
	echo "The requested version $newver and the current version $vrchk2 are the SAME"
	break
else

 if [ "$lvrarg3" >= "$lvrarg4" ]
 then
 for (n=0;n<=$lvararg3;n++)
 do
    if ( ${vrarg4[n]} < ${vrarg3[n]} ) 
    then
        echo "Upgrade required "
        upgradeflag="1"
        #break	
    fi

 done

 fi

 if [ "$lvrarg3" < "$lvrarg4" ]
 then
 for (n=0;n<=$lvararg4;n++)
 do
    if ( ${vrarg4[n]} > ${vrarg3[n]} )
    then
        echo "Upgrade required "
        upgradeflag="1"
        #break
    fi
 done

 fi
  
  if [ $upgradeflag -ne 1 ]
  then
	  echo "Upgrading package"
	  zlibadd
  fi

fi
}

cuda_cuDNN() {

#pre-requistise install zlib getting it from my repo ansible-install source above ...calling the install here

zadd1=`python3 -c "import zlib ; print(zlib.ZLIB_RUNTIME_VERSION)"`
zadd1s="$?"
zlibv="1.3.1"

if [ "$zadd1s" -ne "0" ]
then
	echo "zlib is not installed"
	zlibadd
else

     if [ $zlibv 



	echo "zlib already in the current version hence proceeding with the setup.."
fi

#if you have installed the CUDA tool-kit form this program then you would have had the key-ring & other packages installed
cudnnv="12"
cuDNN=`sudo $cmd1 -y install cudnn-cuda-${cudnnv}`


}

tensorrt_install() {

version="10.x.x.x"
arch=$(uname -m)
cuda="cuda-x.x"
#tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz

#os="ubuntuxx04"
os="${ki}${irelease}"
#tag="10.x.x-cuda-x.x"


#tensorRTv="10.x.x"
#cudaVer="cuda-x.x"

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

alternate-installtensorRT() {
Download:
`tar -xzvf TensorRT-${version}.Linux.${una}-gnu.${cuda}.tar.gz`
export LD_LIBRARY_PATH=<TensorRT-${version}/lib>:$LD_LIBRARY_PATH

cd TensorRT-${version}/python

python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl

}
#checking to see if python and pip are installed before installing cuda for python

chkifinsta python3 pip3

