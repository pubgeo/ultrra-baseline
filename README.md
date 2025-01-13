# Setup
- Install FFMPEG: `sudo apt update` and `sudo apt install ffmpeg`
- Follow instructions [here](https://github.com/nerfstudio-project/nerfstudio?tab=readme-ov-file#1-installation-setup-the-environment) to setup environment and install nerfstudio (this baseline is tested on nerfstudio version 1.1.5, so we recommend installing that with `pip install nerfstudio==1.1.5`)
- Install required packages for baseline with `pip install -r requirements.txt`
- Follow instructions [here](https://github.com/cvg/Hierarchical-Localization?tab=readme-ov-file#installation) to install `hloc`
- Download contest data (`inputs` and possibly `ref` and `res` dirs) into some `root_datasets_dir` location (needed for Usage below).

# Usage
- Example to run View Synthesis stage on dev dataset, `t01_v09_s00_r01_ImageDensity_WACV_dev_A01`
    - `python --root_datasets_dir /PATH/TO/DIR/WITH/INPUTS --stage view_synthesis --dataset_name t01_v09_s00_r01_ImageDensity_WACV_dev_A01`
    - ^This should output results into a `res` folder inside the `root_datasets_dir`
- See `python baseline.py --help` for more information.
- Example run shell scripts are provided.

# Coordinate Systems Alignment
- During the view synthesis stage, the ground truth camera locations are provided in geodetic coordinates (latitude, longitude, and altitude) as specified in the metadata JSON file. These geodetic coordinates can be converted to the world reference East-North-Up (ENU) Cartesian coordinate system, represented by `xEast`, `yNorth`, and `zUp`, using the `geodetic2enu` function.
- For evaluation purposes, the reconstructed coordinates must be aligned with the ENU world coordinate system. The [baseline code](https://github.com/pubgeo/ultrra-baseline/blob/main/baseline.py) provides an example demonstrating how to align reconstructed local coordinates to the ENU coordinates.
    - **COLMAP Reconstructions**: COLMAP may reconstruct multiple models when not all images are registered into a single model. These reconstructed models (`arb_colmap_models`) are stored in the arb_colmap_dir, and the camera locations in local coordinates are saved in `successful_arb_colmap_cart_dicts`.  
    - **Coordinate Alignment**: For each model in arb_colmap_models, a transformation is calculated between the local camera locations and their corresponding ENU world coordinates using the `Procrustes` function. The models are then transformed to the ENU coordinate system using these calculated transformations and saved as `transformed_arb_colmap_models`. 
    - **Model Merging**: Once transformed, the aligned models can be merged into a single model using the `merge_models` function

# Submission to Codabench
- In order to create your zip file that you upload to codabench, please organize your folders into a specific format, otherwise the scoring program will error. The format expected is:
```
submission/
├─ view_synthesis/
│  ├─ t01_v09_s00_r01_ImageDensity_WACV_dev_A01/
│  │  ├─ *.jpg
│  │  ├─ ...
│  ├─ t02_v06_s00_r01_CameraModels_WACV_dev_A01/
│  │  ├─ *.jpg
│  │  ├─ ...
│  ├─ .../
├─ camera_calibration/
│  ├─ t01_v09_s00_r01_ImageDensity_WACV_dev_A01/
│  │  ├─ *.json
│  │  ├─ ...
│  ├─ t02_v06_s00_r01_CameraModels_WACV_dev_A01/
│  │  ├─ *.json
│  │  ├─ ...
│  ├─ .../
```
- The `view_synthesis/` and `camera_calibration` folders represent the separate tracks
- Within these folders are the separate dataset names for each track (for example, `t01_v09_s00_r01_ImageDensity_WACV_dev_A01`)
- Within the folders for each dataset name are your respective submission files. For `camera_calibration/` these will be JSONs. For `view_synthesis/`, these will be JPGs. 
- **When submitting to Codabench, DO NOT zip a high level folder containing the `view_synthesis/` and `camera_calibration/` folders. Zip each folder individually, otherwise the scoring program will error.** For example:
    - **CORRECT:** `zip -r my-submission.zip view_synthesis/ camera_calibration/`
    - **INCORRECT:** `zip -r my-submission.zip submission/`
