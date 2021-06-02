#!/bin/bash

srun \
    -K --gpus=1 \
    --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh \
    --container-workdir="$(pwd)" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
    python main.py
