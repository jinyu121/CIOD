#!/bin/sh
echo "CUDA_VISIBLE_DEVICES=XXX ./runall.sh SESSION_NUM"
SESSION=$1

cmd="python3 trainval_net.py --use_tfboard --session ${SESSION}"
echo $cmd
$cmd > logs/${SESSION}_train.log
echo $cmd

cmd="python3 test_net.py --s ${SESSION} --no_repr"
echo $cmd
$cmd > logs/${SESSION}_test-norepr.log
tail -n11 logs/${SESSION}_test-norepr.log
echo $cmd
