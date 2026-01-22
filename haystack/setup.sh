#!/usr/bin/env bash
#author:rangapv@yahoo.com
#22-01-26

prep(){

hs1=`sudo mkdir -p /ephemeral/temp`
hs2=`sudo chmod 777 /ephemeral/temp`
hs4=`echo "export PYTORCH_ALLOC_CONF=expandable_segments:True" >> $HOME/.bashrc`
hs5=`echo "export HF_HUB_CACHE=/ephemeral/temp" >> $HOME/.bashrc`
}

prep
