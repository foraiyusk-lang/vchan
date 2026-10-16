#!/bin/bash
# usage: bench_speed.sh <tag> <bs> [ENV=VAL ...]  -> prints epoch seconds (16-step epochs)
cd /c/Users/ysk72/reimu_ai_vc
tag=$1; bs=$2; shift 2
for kv in "$@"; do export "$kv"; done
rm -f logs/bench_train.log
.venv/Scripts/python.exe scripts/run_training.py bench data/train_v1 --stages train --epochs 3 --save-every 100 --batch $bs > logs/bench_$tag.log 2>&1
echo "== $tag bs=$bs $@"; grep "====>" logs/bench_train.log
