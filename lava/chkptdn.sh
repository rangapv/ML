#!/usr/bin/env bash
#auhtor:rangapv@yahoo.com
#20-05-2026


chkpt_dwn(){

lavatmp="lava-temp1/LLaVA"
#lavatmp="/home/ubunut/LLaVA-1.1.3"
#chk0=`pip3 install --upgrade huggingface_hub`
chk1=`hf auth login --no-add-to-git-credential`

env1=`export HF_HUB_DISABLE_SYMLINKS=1`
env2=`export HF_HUB_DISABLE_SYMLINKS_WARNING=True`
chk2=`hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./$lavatmp/checkpoints/Meta-Llama-3-8B-Instruct`

chk3=`hf download openai/clip-vit-large-patch14-336 --local-dir ./$lavatmp/checkpoints/clip-vit-large-patch14-336`

chk4=`hf download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5 --local-dir ./$lavatmp/checkpoints/llava-pretrain-projector`

chk5=`hf download lmsys/vicuna-7b-v1.5 --local-dir ./checkpoints/vicuna-7b-v1.5 --local-dir ./$lavatmp/checkpoints/ `

chk5=`cd ./$lavatmp/playground/data; wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json`

}

chkpt_dwn
