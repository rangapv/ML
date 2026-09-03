#!/usr/bin/env bash
#author:rangapv@yahoo.com
#03-09-2026

ins_3dgrut(){

ins1=`git clone https://github.com/nv-tlabs/3dgrut.git --recursive`
ins2=`cd ./3dgrut;source .venv/bin/activate;./install_env_uv.sh`

}

sys_prep(){

sys1=`sudo apt install python3.11`
sys2=`sudo ln -sf ./python3.11 ./python3`
sys3=`pip3 install -r ./requirements.txt`

}

sys_prep
ins_3dgrut
