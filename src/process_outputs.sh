#!/bin/bash

echo "Compute centroids"
CROPPED_DATASETS="$(find . -type d -iname "data_*")"
for d in ${CROPPED_DATASETS[@]}; do
    echo "Compute centroids from dataset: $d"
    python centroids.py ${d}
done

echo "Plot centoids"
CENTROIDS_FILES="$(find ../res/ -type f -iname "centroids_*.json")"
for c in ${CENTROIDS_FILES[@]}; do
    echo "Ploting centroid file ${c}"
    python centroid_plot.py ${c}
done

echo "Create charts"
python subvales_plot.py ../res/results.json 
