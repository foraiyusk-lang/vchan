#!/bin/bash
# Stage 1: CUDA 11.8 torch pair (GTX 1650 = pre-RTX50 -> cu118); Stage 2: remaining deps
cd /c/Users/ysk72/reimu_ai_vc
set -x
uv pip install --python .venv/Scripts/python.exe torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
echo "STAGE1 exit=$?"
uv pip install --python .venv/Scripts/python.exe -r requirements_cu118_official.txt
echo "STAGE2 exit=$?"
uv pip install --python .venv/Scripts/python.exe huggingface_hub fugashi unidic-lite requests
echo "STAGE3 exit=$?"
