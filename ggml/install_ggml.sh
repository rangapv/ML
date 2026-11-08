#!/usr/bin/env bash
#author:rangapv@yahoo.com
#06-11-2026

insta_dep() {

dep1=`sudo apt update` 
dep2=`sudo apt install -y build-essential cmake git pkg-config`

}

insta_ggml() {
insggm1=`git clone https://github.com/ggml-org/ggml`
insggm2=`cd ggml;mkdir build && cd build;cmake ..;cmake --build . --config Release -j 8`
insggm3=``

}

complie-sample() {
samp1=`cd /ggml/build ;cmake --build . --target simple-backend -j2`
}

insta_dep

insta_ggml

compile-sample

