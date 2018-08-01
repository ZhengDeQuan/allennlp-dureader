#!/bin/bash
#source /mnt/lustre/share/miniconda3/envsetup9.0.sh
#source activate r0.1.1
which python

datadir=/mnt/lustre/share/wangchangbao/datasets/MRC2018/preprocessed
search_train_file=${datadir}/trainset/search.train.json
zhidao_train_file=${datadir}/trainset/zhidao.train.json
search_dev_file=${datadir}/devset/search.dev.json
zhidao_dev_file=${datadir}/devset/zhidao.dev.json
combine_dev_file=${datadir}/devset/combine.dev.json.fake_ref
search_test1_file=${datadir}/test1set/search.test1.json
zhidao_test1_file=${datadir}/test1set/zhidao.test1.json


ROOT=../..
export PYTHONPATH=${ROOT}:$PYTHONPATH
log=eval.log
pred_file=$1
ref_file=${datadir}/devset/$2.dev.json.fake_ref
pred_file=/mnt/lustre/share/wangchangbao/datasets/MRC2018/preprocessed/trainset/search.train.json.fake_pred
ref_file=/mnt/lustre/share/wangchangbao/datasets/MRC2018/preprocessed/trainset/search.train.json.fake_ref
echo "--------------------------------------------------" >> $log
echo $pred_file >> $log
python $ROOT/tools/dureader_eval.py ${pred_file} ${ref_file} Main | tee -a $log
