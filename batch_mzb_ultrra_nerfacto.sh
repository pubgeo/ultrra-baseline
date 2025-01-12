#!/bin/bash

# For ULTRRA dev datasets
DATASET_DIR=/data/home/brownmz1/ultrra-challenge/dev_contest_datasets_241219_reduced_2
METHOD=nerfacto
RESULTS_NAME=res_nerfacto

# this is a hack, but keep running this until COLMAP stops aborting
# the code is written to check for valid output and skip if found
for i in {1..1000}
do
    echo "Try #$i"
    python ultrra-baseline/baseline.py --root_datasets_dir $DATASET_DIR --stage camera_calibration --output_name $RESULTS_NAME
    python ultrra-baseline/baseline.py --root_datasets_dir $DATASET_DIR --stage view_synthesis --method_to_use $METHOD --output_name $RESULTS_NAME
done

# For ULTRRA test datasets
DATASET_DIR=/data/home/brownmz1/ultrra-challenge/test_contest_datasets_241219_reduced_2
METHOD=nerfacto
RESULTS_NAME=res_nerfacto

# this is a hack, but keep running this until COLMAP stops aborting
# the code is written to check for valid output and skip if found
for i in {1..1000}
do
    echo "Try #$i"
    python ultrra-baseline/baseline.py --root_datasets_dir $DATASET_DIR --stage camera_calibration --output_name $RESULTS_NAME
    python ultrra-baseline/baseline.py --root_datasets_dir $DATASET_DIR --stage view_synthesis --method_to_use $METHOD --output_name $RESULTS_NAME
done

