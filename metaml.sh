#!/bin/bash
set -E
source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/main/pyverchk.sh) >/dev/null 2>&1
#source "../ans/pyverchk.sh"

depflag1=0

pyins() {

        source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/main/py.sh) 

}

pipins() {

        source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/main/pipak.sh)
	pip install gdown

}

mlnvidia() {

	nvc=`nvcc -V`
	snvc="$?"
        statuschk snvc
	gcc=`gcc --version`
	sgcc="$?"
        statuschk sgcc

}

statuschk() {
sck1="$@"

if [[ (( $sck1 -ne 0 )) ]]
then
	echo "Dependency $sck1 not installed"
        depflag1=1
fi

}


#Main Begins

pythonwc
if [[ ( "$pyvs" !=  "0" ) ]]
then
   echo "inside pyvs"
   pyins
   pipins
fi

mlnvidia

if [[ (( $depflag1 -ne 0 )) ]]
then
	ndriv=`sudo apt install -y nvidia-driver-460`
	sndriv="$?"
        nvidtool=`sudo apt-get install -y nvidia-cuda-toolkit`
        snvidtool="$?"
	nvidprim=`sudo apt install -y nvidia-prime`
	snvidprim="$?"
	wnvid=`whereis nvidia`
	swnvid="$?"
        echo "The Nvidia is installed in $wnvid"	
        whichprim=`prime-select query`
        swhichprim="$?"
	echo "The GPU selected is $whichprim"
        hardware=`sudo lshw -C display`
	shardware="$?"
	echo "The hardware features are $hardware"
else
	echo "Nvidia Requirements Met..."
	exit
fi
	
nvidsmi=`nvidia-smi`
snvidsmi="$?"
if [[ (( $snvidsmi -eq 0 )) ]]
then
        echo "Nvidia smi is $nvidsmi"
	echo "It is all good to go!"
else
	echo "Nvidia smi is not working"
fi

