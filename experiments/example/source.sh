#!/bin/bash
source /mnt/lustre/share/miniconda3/envsetup.sh
source /mnt/lustre/share/wangchangbao/workspace/py3torch/bin/activate
which python

datadir=/mnt/lustre/share/wangchangbao/datasets/MRC2018/preprocessed
search_train_file=${datadir}/trainset/search.train.json
zhidao_train_file=${datadir}/trainset/zhidao.train.json
search_dev_file=${datadir}/devset/search.dev.json
zhidao_dev_file=${datadir}/devset/zhidao.dev.json
search_test1_file=${datadir}/test1set/search.test1.json
zhidao_test1_file=${datadir}/test1set/zhidao.test1.json


ROOT=../..
export PYTHONPATH=${ROOT}:$PYTHONPATH

