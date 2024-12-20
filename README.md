# Setup
- Install FFMPEG: `sudo apt update` and `sudo apt install ffmpeg`
- Follow instructions [here](https://github.com/nerfstudio-project/nerfstudio?tab=readme-ov-file#1-installation-setup-the-environment) to setup environment and install nerfstudio (this baseline is tested on nerfstudio version 1.1.5, so we recommend installing that with `pip install nerfstudio==1.1.5`)
- Install required packages for baseline with `pip install -r requirements.txt`
- Follow instructions [here](https://github.com/cvg/Hierarchical-Localization?tab=readme-ov-file#installation) to install `hloc`
- Download contest data (`inputs` and possibly `ref` and `res` dirs) into some `root_datasets_dir` location (needed for Usage below).

# Usage
- (Example to run View Synthesis stage on dev dataset, `t01_v09_s00_r01_ImageDensity_WACV_dev_A01`)
    - `python --root_datasets_dir /PATH/TO/DIR/WITH/INPUTS --stage view_synthesis --dataset_name t01_v09_s00_r01_ImageDensity_WACV_dev_A01`
    - ^This should output results into a `res` folder inside the `root_datasets_dir`
- See `python baseline.py --help` for more information.


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
