#!/bin/bash

GPU_BOARD="GTX1080Ti"
GPU_NUM=1
MYSELF=${0##*/}:w
MEMORY="50GB"

function help()
{
usage=$(cat << EOF
Usage: $MYSELF [OPTIONS]
  Description:
       This script helps to launch the container with all dependencies
       required to execute the ML experiment. In addition, it includes
       some extra options that helps to debug and clean environment.
  Options:
       -h,    Help page.
       -i,    Run container in interactive model.
       -p,    Extract information about centroids and plot all
              information from JSON files.
       -c,    Clean plots and centroids files.
       -a,    Clean all.
EOF
)
	echo "$usage";
}


if [ $# -eq 0 ]; then
    echo "Execute container"
    srun \
        -K --gpus=$GPU_NUM \
        -p "$GPU_BOARD" \
        --mem="$MEMORY" \
        --container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh \
        --container-workdir="$(pwd)" \
        --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
        bash job.sh
    exit 0
else
    while getopts ":hcip" opt; do
        case ${opt} in
        h)
            help
            exit 0
        ;;
        a)
            echo "Clean outputs"
            set -x
            rm -rf ../res/*
            rm -rf ../exp
            rm -rf *.png
            rm -rf *.csv
            rm -rf data_*
            rm -f results.json
            rm -f PAAI21_CIFAR10_model.pt
            rm -rf __pycache__
            exit 0
        ;;
        c)
            rm -rf ../res/centroids_data_*
            rm -rf ../res/centroids_charts
            rm -rf ../res/charts
        ;;
        i)
            echo "interactive mode"
            srun \
                -K --gpus=$GPU_NUM \
                -p "$GPU_BOARD" \
                --mem="$MEMORY" \
                --container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh \
                --container-workdir="$(pwd)" \
                --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
                --pty bash intv.sh
            exit 0
        ;;
        p)
            echo "Plot results and centroids"
            srun \
                -K --gpus=$GPU_NUM \
                -p "$GPU_BOARD" \
                --container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh \
                --container-workdir="$(pwd)" \
                --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
                --pty bash process_outputs.sh

            echo "Plot everything"
        ;;
        \?)
            help
            exit 0
        ;;
        esac
done
fi
