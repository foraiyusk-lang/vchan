#!/bin/bash
cd /c/Users/ysk72/reimu_ai_vc/rvc
HF=/c/Users/ysk72/reimu_ai_vc/.venv/Scripts/hf.exe
set -x
$HF download lj1995/VoiceConversionWebUI --revision main --include "hubert_base/*" --local-dir assets
$HF download lj1995/VoiceConversionWebUI rmvpe.pt --revision main --local-dir assets/rmvpe
$HF download lj1995/VoiceConversionWebUI --revision main --include "pretrained_v2/*" --local-dir assets
$HF download lj1995/VoiceConversionWebUI mute.zip --revision main --local-dir .model-downloads
/c/Users/ysk72/reimu_ai_vc/.venv/Scripts/python.exe -m zipfile -e .model-downloads/mute.zip logs
echo "ASSETS DONE exit=$?"
