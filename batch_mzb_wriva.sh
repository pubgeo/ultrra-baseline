#!/bin/bash

# For WRIVA datasets
DATASET_DIR=/home/brownmz1/wriva-challenge-datasets-ultrra
METHOD=nerfacto
RESULTS_NAME=res

# this is a hack, but keep running this until COLMAP stops aborting
# the code is written to check for valid output and skip if found
for i in {1..1000}
do
    echo "Try #$i"
    python ultrra-baseline/baseline.py --root_datasets_dir $DATASET_DIR --stage camera_calibration --output_name $RESULTS_NAME
    python ultrra-baseline/baseline.py --root_datasets_dir $DATASET_DIR --stage view_synthesis --method_to_use $METHOD --output_name $RESULTS_NAME
done

