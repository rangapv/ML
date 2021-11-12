#!/bin/bash
set -E
source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/main/pyverchk.sh) >/dev/null 2>&1


depflag1=0

pyins() {

        source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/main/py.sh) >/dev/null 2>&1

}

pipins() {

        source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/main/pipak.sh) >/dev/null 2>&1

}

mlnvidia() {

	pip install gdown
	nvc=$(nvcc -V)
	snvc="$?"
        statuschk snvc
	gcc=$(gcc --version)
	sgcc="$?"
        statuschk sgcc


}

statuschk() {
sck1="$?"

if [[ ($sck1 -ne 0 )) ]]
then
	echo "Dependency $sck1 not installed"
        depflag1=1
fi

}

pythoncurrent
if [[ ( $pyvs -ne 0 ) ]]
then
   echo "No Python Found; GREENFIELD Installs proceeding"
   pyins
   pipins
fi

mlnvidia

if [[ (( $depflag -eq 0 )) ]]
then
	ndriv=`sudo apt install nvidia-driver-460`
	sndriv="$?"
        nvidtool=`sudo apt-get install nvidia-cuda-toolkit`
        snvidtool="$?"
	wnvid=`whereis nvidia`
	swnvid="$?"
	nvidsmi=`nvidia-smi`
	snvidsmi="$?"
fi

