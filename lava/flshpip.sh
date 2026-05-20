#!/usr/bin/env bash
#author:rangapv@yahoo.com
#20-05-2026

#source <(curl -s https://raw.githubusercontent.com/rangapv/CloudUtil/refs/heads/main/awsconfig.sh) > /dev/null 2>&1

lavatmp="lava-temp1"
cm1=`mkdir $lavatmp`
cm2=`cd $lavatmp;git init;git clone https://github.com/haotian-liu/LLaVA.git`
cm4=`cd $lavatmp/LLaVA;pip3 install --upgrade pip`
cm5=`cd $lavatmp/LLaVA;pip3 install -e .`
cm6=`cd $lavatmp/LLaVA;pip3 install -e ".[train]"`
cm7=`pip3 install --upgrade torch==2.11.0`
cm70=`curl -s https://s3.us-east-2.amazonaws.com/flash-attn-2.8.3/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl -o ./flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl`
#cm71=`aws configure`
#cm73=`aws s3 cp s3://flash-attn-2.8.3/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl`
cm71=`pip3 install ./flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl`

