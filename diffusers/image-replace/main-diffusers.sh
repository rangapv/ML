#!/usr/bin/env bash
#author:rangapv@yahoo.com
#25-01-26

install_diff() {

id1=`git clone https://github.com/huggingface/diffusers.git`
id2=`cd ./diffusers`
id3=`pip3 install -e ".[torch]"`
id4=`echo "export PATH=/home/ubuntu/d1/diffusers:$PATH" >> $HOME/.bashrc` 

}

install_diff
