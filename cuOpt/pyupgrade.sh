#!/usr/bin/env bash
#author:rangapv@yahoo.com
#01-02-26

pi1=`sudo add-apt-repository ppa:deadsnakes/ppa`
pi10=`sudo apt update`
pi11=`sudo apt-get install python3.11`
pi12=`sudo ln -sf /usr/bin/python3.11 /usr/bin/python3`
pip13=`pip3 install -r ./requirements.txt`
