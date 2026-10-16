#!/bin/bash
# Keeps `reimu_v1` training alive: if the trainer died (e.g. Windows paging-file errors),
# relaunch it; RVC resumes from the latest G_/D_ checkpoint. Exits once the final weights exist.
cd /c/Users/ysk72/reimu_ai_vc
FINAL=rvc/assets/weights/reimu_v1.pth
n=0
while [ ! -f "$FINAL" ]; do
  if ! powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -like 'python*' -and \$_.CommandLine -like '*-P train/train.py*' }) { exit 0 } else { exit 1 }"; then
    n=$((n+1)); echo "$(date) trainer not running -> relaunch #$n"
    if [ $n -gt 12 ]; then echo "too many restarts"; exit 1; fi
    sleep 20
    .venv/Scripts/python.exe scripts/run_training.py reimu_v1 data/train_v1 --stages train --epochs 40 --save-every 2 --batch 1 > logs/train_v1_resume$n.log 2>&1 &
    sleep 240
  fi
  sleep 120
done
echo "$(date) final weights exist: $FINAL"
