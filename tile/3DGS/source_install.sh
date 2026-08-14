#!/usr/bin/env bash
#author:rangapv@yahoo.com
#14-08-2026

wheel_install(){

wi1=`pip3 install --pre torch==2.13.0 torchvision==0.28.0`

wi2=`pip3 install --user --upgrade packaging setuptools`

wi3=`pip3 install tyro`

si1=`sudo apt install unzip`

si2=`pip3 uninstall -y tensorflow keras tf-keras tensorflow-io-gcs-filesystem`

wi4=`wget https://s3.us-east-2.amazonaws.com/gsplat-1.5.3/gsplat-1.5.3-cp310-cp310-linux_x86_64.whl`

wi5=`pip3 install ./gsplat-1.5.3-cp310-cp310-linux_x86_64.whl`


#TEST:

python3 -c "import gsplat; print(gsplat.__file__); from gsplat import color_correct; print('ok')"



wi5=`pip3 install nerfview`


si3=`mkdir sample1;cd sample1;git init;git pull https://github.com/nerfstudio-project/gsplat.git`

si4=`cd ./sample1/examples; python3 datasets/download_dataset.py`

si5=`cd ./sample1/examples;bash benchmarks/basic.sh`


}


wheel_install
