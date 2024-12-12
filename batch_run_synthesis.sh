#!/bin/bash

# Change to the directory containing the Python script
cd /home/lig1

# Change to the directory containing the Python script
conda activate nerfstudio

## Run the Python script
python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/dev_phase/dev_contest_datasets_241211/ \
    --dataset_name t01_v09_s00_r01_ImageDensity_WACV_dev_A01 \
    --cuda_visible_devices 0

python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/dev_phase/dev_contest_datasets_241211/ \
    --dataset_name t02_v06_s00_r01_CameraModels_WACV_dev_A01 \
    --cuda_visible_devices 0

python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/dev_phase/dev_contest_datasets_241211/ \
    --dataset_name t03_v06_s00_r01_ReconstructedArea_WACV_dev_A01 \
    --cuda_visible_devices 0

python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/dev_phase/dev_contest_datasets_241211/ \
    --dataset_name t04_v11_s00_r01_VaryingAltitudes_WACV_dev_A01 \
    --cuda_visible_devices 0


# Run the Python script
python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/test_phase/test_contest_datasets_241211/ \
    --dataset_name t01_v10_s00_r01_ImageDensity_WACV_test_A09 \
    --cuda_visible_devices 0


python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/test_phase/test_contest_datasets_241211/ \
    --dataset_name t02_v07_s00_r01_CameraModels_WACV_test_A09 \
    --cuda_visible_devices 0


python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/test_phase/test_contest_datasets_241211/ \
    --dataset_name t03_v07_s00_r01_ReconstructedArea_WACV_test_A09 \
    --cuda_visible_devices 0

python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/test_phase/test_contest_datasets_241211/ \
    --dataset_name t04_v12_s00_r01_VaryingAltitudes_WACV_test_A09 \
    --cuda_visible_devices 0

# Run the Python script
python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/test_phase/test_contest_datasets_241211/ \
    --dataset_name t01_v11_s00_r01_ImageDensity_WACV_test_A10 \
    --cuda_visible_devices 0


python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/test_phase/test_contest_datasets_241211/ \
    --dataset_name t02_v08_s00_r01_CameraModels_WACV_test_A10 \
    --cuda_visible_devices 0


python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/test_phase/test_contest_datasets_241211/ \
    --dataset_name t03_v08_s00_r01_ReconstructedArea_WACV_test_A10 \
    --cuda_visible_devices 0


python3 ultrra-baseline/baseline.py --stage view_synthesis \
    --root_datasets_dir /media/wriva/Data/WACV25/test_phase/test_contest_datasets_241211/ \
    --dataset_name t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10 \
    --cuda_visible_devices 0

