#!/usr/bin/env bash
#author:rangapv@yahoo.com
#10-05-26

#Reference: https://github.com/haotian-liu/LLaVA

fix1(){

lavatmp="lava-temp1"
cm1=`mkdir $lavatmp`
cm11=`mkdir $lavatmp/wheels`
cm2=`cd $lavatmp;git init;git clone https://github.com/haotian-liu/LLaVA.git`
cm4=`cd $lavatmp/LLaVA;pip3 install --upgrade pip`
cm5=`cd $lavatmp/LLaVA;pip3 install -e .`
cm6=`cd $lavatmp/LLaVA;pip3 install -e ".[train]"`
cm7=`pip3 install --upgrade torch==2.11.0`
cm8=`pip3 install --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu132`
cm9=`cd $lavatmp/LLaVA;MAX_JOBS=4 pip3 wheel flash-attn -w ../wheels/ --no-build-isolation`
#cm7=`cd $lavatmp/LLaVA;MAX_JOBS=4 pip3 install flash-attn --no-build-isolation --no-cache-dir`

}

fix1
