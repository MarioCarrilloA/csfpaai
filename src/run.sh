#!/bin/bash

srun \
    -K --gpus=1 \
    --mem=50GB \
    --container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh \
    --container-workdir="$(pwd)" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
    python main.py
