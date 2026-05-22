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
cm71=`pip3 install --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu132`
cm70=`curl -s https://s3.us-east-2.amazonaws.com/flash-attn-2.8.3/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl -o ./flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl`
#cm71=`aws configure`
#cm73=`aws s3 cp s3://flash-attn-2.8.3/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl`
cm71=`pip3 install ./flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl`


chkpt_dwn(){

chk0=`pip3 install --upgrade huggingface_hub`
chk1=`hf auth login --no-add-to-git-credential`

env1=`export HF_HUB_DISABLE_SYMLINKS=1`
env2=`export HF_HUB_DISABLE_SYMLINKS_WARNING=True`
chk2=`hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./$lavatmp/LLaVA/checkpoints/Meta-Llama-3-8B-Instruct`

chk3=`hf download openai/clip-vit-large-patch14-336 --local-dir ./$lavatmp/LLaVA/checkpoints/clip-vit-large-patch14-336`

chk4=`hf download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5 --local-dir ./$lavatmp/LLaVA/checkpoints/llava-pretrain-projector`

chk5=`hf download lmsys/vicuna-7b-v1.5 --local-dir ./checkpoints/vicuna-7b-v1.5 --local-dir ./$lavatmp/LLaVA/checkpoints/ `

chk5=`cd ./playground/data; wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json`

}

chkpt_dwn
