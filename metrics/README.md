# ULTRRA Baseline Metrics

This sub folder contains the metrics code that is utilized on Codabench to run evaluations. We also provide a 
simple visualization script that allows you to compare the scores across the different datasets.

## Installation

Utilize our `requirements.txt` in order to install the dependencies needed for running locally:
```bash
pip install -r requirements.txt
```

## Running

Simply call the `evaluate.py` script on the folder that contains your `ref` and `res` folders as well as an output folder.
As long as your folder structure is correct, the evaluation code should run without issue. See below for the expected format:

```bash
python3 evaluate.py /path/to/submission/folder /path/to/output/folder
```

This produces a `scores.txt` which contains the average score for each metric. This is what would be displayed on the leaderboard.
It also produces a `per_frame_result.json`, which contains frame-level scores for both camera calibration and view synthesis
for each dataset.

After running `evaluate.py`, you can visualize the scores with `visualize.py`. To create the plots, call the script, passing
in the path to the `per_frame_result.json` that was produced earlier:

```bash
python3 visualize.py /path/to/output/folder/per_frame_result.json
```
### Expected Format

This is a reiteration from the README on the top level of this baseline repo, but ensure your format that you use for
both submission and running this metrics code is correct:

In order to create your zip file that you upload to codabench, please organize your folders into a specific format, otherwise the scoring program will error. The format expected is:
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

<!-- # wriva-codabench

Codabench format for the WRIVA public challenge. 

## Usage

I use makefiles in order to create the proper format that codabench expects. 

To make the competition run (in the project root):

```
make
```

This creates a `wriva-codabench-competition.zip` file that you can upload directly to codabench

To clean up the competition artifacts, run:

```
make clean
```


## Overview

The ULTRRA challenge evaluates current and novel state of the art view synthesis methods for posed and unposed cameras. Challenge datasets emphasize real-world considerations, such as image sparsity, variety of camera models, and unconstrained acquisition in real-world environments. 

Schedule:
- Development dataset release: 11/8/2024
- Challenge dataset release: 1/10/2025
- Submission period: 11/1/2024 - 1/31/2025
- Invited presentations for selected participants: 2/14/2025

To get started, please register for the competition and download the development data package from IEEE DataPort (https://ieee-dataport.org/competitions/ultrra-challenge-2025).

Tasks:
- Camera calibration
  - Inputs: unposed images
  - Ouputs: relative camera extrinsic parameters
  - Evaluation: camera geolocation error
- View synthesis
  - Inputs: images with camera locations, camera metadata for requested novel image renders
  - Outputs: rendered images
  - Evaluation: DreamSim image similarity metric

Challenges posed for each task, increasing in complexity:
- Image density: a limited number of input images from a single ground-level camera
- Multiple camera models: images from multiple unique ground-level cameras
- Varying altitudes: images from ground, security, and airborne cameras
- Reconstructed area: images from varying altitudes, covering a larger area

Example datasets are provided for each task and challenge to support algorithm development. An example baseline solution is provided based on COLMAP and NerfStudio, and a baseline submission is provided to clarify the expected submission format.

## Data

Images collected for the IARPA WRIVA program have been publicly released and made available for use in this public challenge and more broadly to encourage research in view synthesis methods for real-world environments and heterogeneous cameras. Datasets include images collected from mobile phones and other ground-level cameras, security cameras, and airborne cameras. Each camera is calibrated using structure from motion constrained by RTK-corrected GPS coordinates, with accuracies measured in centimeters, for either camera locations or ground control points, depending on the camera. Cameras are geolocated to enable reliable evaluation. Images used for final evaluations are sequestered.

Development datasets are available for download at IEEE DataPort (https://ieee-dataport.org/competitions/ultrra-challenge-2025), and the challenge datasets will be posted there 10 January 2025. 

## Evaluation

During the development phase, both camera calibration and view synthesis tasks will be evaluated using images from the same scene. Reference values for both tasks are provided in the development data package, allowing participants to  independently experiment and self-evaluate and also confirm that submissions to the leaderboard are formatted correctly.

During the test phase, camera calibration and view synthesis tasks will be evaluated using images from different scenes, and all reference values will be sequestered.

Camera calibration task:
Inputs are unposed JPG images, and outputs are relative camera poses in a JSON file format. Camera poses are evaluated by comparing relative camera locations, with contestant coordinate frames aligned to sequestered reference world coordinates using Procrustes analysis. The leaderboard metric for this task is SE90, the 90th percentile spherical error from all input images.

View synthesis task:
Inputs are posed images for training a view synthesis model and test poses for rendering novel views. Sequestered images and contestant rendered images will be compared using the single-model variant of the DreamSim image similarity metric (https://dreamsim-nights.github.io/). The leaderboard metric for this task is the mean DreamSim score from all rendered images.

## Terms

By participating in this challenge, you consent to publicly share your submissions.

No prizes will be awarded. Organizers will review submissions and invite top-scoring participants to submit a brief writeup of their solutions, documenting their approaches and observations. Organizers will then select from among the best scores and writeups and invite participants to present their results at the ULTRRA workshop at WACV 2025 (https://sites.google.com/view/ultrra-wacv-2025). Virtual presentations will be supported. All top-scoring participants on the leaderboard will be acknowledged at the workshop.

## Acknowledgements

This work was supported by the Intelligence Advanced Research Projects Activity (IARPA) contract no. 2020-20081800401. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA or the U.S. Government.

 -->
