#!/bin/bash
#source /mnt/lustre/share/miniconda3/envsetup9.0.sh
#source activate r0.1.1
#source /mnt/lustre/share/wangchangbao/workspace/py3torch/bin/activate
which python

datadir=/mnt/lustre/share/wangchangbao/datasets/MRC2018/preprocessed
search_train_file=${datadir}/trainset/search.train.json
zhidao_train_file=${datadir}/trainset/zhidao.train.json
search_dev_file=${datadir}/devset/search.dev.json
zhidao_dev_file=${datadir}/devset/zhidao.dev.json
search_test1_file=${datadir}/test1set/search.test1.json
zhidao_test1_file=${datadir}/test1set/zhidao.test1.json
mini_train_file=${datadir}/trainset/mini.train.json
mini_dev_file=${datadir}/devset/mini.dev.json


ROOT=../..
#export PYTHONPATH=${ROOT}:$PYTHONPATH
export PYTHONPATH=${ROOT}:/mnt/lustre/share/wangchangbao/workspace/py3torch/lib/python3.6/site-packages:$PYTHONPATH

    python -W ignore ${ROOT}/tools/train_val.py \
    -e \
    -j 2 \
    --batch-size=2 \
    --warmup_epochs=1 \
    --step_epochs=5 \
    --epochs=10 \
    --lr=0.001 \
    --results_dir=results_zhidao \
    --model_dir=models \
    --vocab_dir=vocab \
    --train_files ${search_train_file} ${zhidao_train_file} \
    --val_files ${zhidao_dev_file} ${search_dev_file} \
    --config=bidaf.json \
    --port=23333 \
    --resume=models/checkpoint_e10.pth \
    2>&1 | tee test.log 

