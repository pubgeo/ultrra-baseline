# ULTRRA Baseline Metrics

(WIP)


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
