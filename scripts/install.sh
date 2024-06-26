#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ..

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
