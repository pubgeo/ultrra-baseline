"""
Evaluation code for ULTRRA Challenge
"""

#!/usr/bin/env python
from collections import defaultdict
import sys
import os
from PIL import Image
import numpy as np
import json
from glob import glob
import time
import datetime
from pathlib import Path
import traceback

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from projections import geodetic_to_enu
from procrustes import procrustes

sglob = lambda p: sorted(glob(p))

print(f"CURRENT TIME: {datetime.datetime.now().time()}")

print("importing dreamsim")
from dreamsim import dreamsim
model, preprocess = dreamsim(pretrained=True, device='cpu', dreamsim_type='open_clip_vitb32')
print("finished importing dreamsim")


def load_metadata(path):
    """
    Loads metadata jsons at path

    :param path: The path that contains metadata jsons
    :return: _description_
    """
    all_json_paths = sglob(os.path.join(path, '*.json'))
    all_metadata = []
    for json_path in all_json_paths:
        with open(json_path, 'r') as f:
            all_metadata.append(json.load(f))
    return all_metadata


def load_images(path):
    """
    Loads images from a path (that are jpgs)

    :param path: The path to the images
    :return: A dictionary of the image name and PIL image
    """
    all_jpgs_paths = sglob(os.path.join(path, "*.jpg"))
    return {Path(img_path).name : Image.open(img_path) for img_path in all_jpgs_paths}


def collect_metadata(predicted_metadata_path):
    """
    Collects all the metadata and a list of indices for images not labeled successful

    :param predicted_metadata_path: The location of all the predicted metadatas
    :return: All the metadata loaded as dictionaries and indices where "success" field is false to ignore in procrustes
    """
    all_metadata = load_metadata(predicted_metadata_path)
    failed_indices = [i for i, metadata in enumerate(all_metadata) if not metadata.get("success")]
    return all_metadata, failed_indices


def extract_coordinates(metadata, keys=("x", "y", "z"), under=None):
    def get_values(d, keys, under):
        source = d.get(under, d) if under else d
        return [source.get(key) for key in keys]
    coords = np.array([[get_values(d, keys, under)] for d in metadata])
    return np.squeeze(coords, axis=1)
        


def evaluate_image(reference, render):
    """
    Evaluates a single image pair. Assumes these image paths are indeed matched

    :param ref_path: path to a reference image that can be opened by pillow
    :param ren_path: path to rendered image that can be opened by pillow
    :return: _description_
    """
    test_image = preprocess(render)
    ref_image = preprocess(reference)
    score = model(ref_image, test_image).item()
    return score


def evaluate_camera_calibration(submission_path, reference_path):
    """
    Evaluates camera calibration competition vector. Given the submission and reference path. 
    First loads submission and reference metadata, removing from the reference those metadatas that failed. We then convert our reference
    LLA to ENU coordinates, which we then use to calculate our transform using the procrustes method. We then apply this transform to submission
    coordinates, and calculate element wise RMSE from that

    :param submission_path: The user-defined submissions folder that contains the JSONs with xyz's
    :param reference_path: The reference data that contains lat lon and alt for the JSONs
    :return: A list of RMSEs between valid submissions
    """
    # returns the metadata as well as the indices of the ones that were not successful
    submission_metadata, failed_indices = collect_metadata(submission_path)
    submission_coordinates = extract_coordinates(submission_metadata)

    reference_metadata = load_metadata(reference_path)
    reference_coordinates = extract_coordinates(reference_metadata, keys=("lat", "lon", "alt"), under="extrinsics")
    assert len(submission_coordinates) == len(reference_coordinates), "The count of reference and submission coordinates do not match!"

    # if reference JSON metadata indicates inaccurate camera locations, then do not use for procrustes fit or camera location evaluation
    # this is not an issue for ULTRRA Challenge datasets
    # it is an issue for some of the WRIVA challenge datasets for anyone converting those to run with ULTRRA solutions and metrics
    keepers = []
    for params in reference_metadata:
        keep = True
        if "geolocation" in params:
            if params["geolocation"] != "gcp" and params["geolocation"] != "rtk" and params["geolocation"] != "synthetic":
                keep = False
        keepers.append(keep)
    reference_coordinates = reference_coordinates[keepers]
    submission_coordinates = submission_coordinates[keepers]

    # call geodetic_to_enu to convert our lat/lon/alt to xyz, collect into numpy array like above
    lat_origin, lon_origin, alt_origin = reference_coordinates[0]
    reference_coordinates = np.array([
        geodetic_to_enu(lat, lon, alt, lat_origin, lon_origin, alt_origin) 
        for lat, lon, alt in reference_coordinates
    ])
    # Get reduced list of coordinates for procrustes fit (and only for fit)
    filtered_reference_coordinates = np.array([coords for i, coords in enumerate(reference_coordinates) if i not in failed_indices])
    filtered_submission_coordinates = np.array([coords for i, coords in enumerate(submission_coordinates) if i not in failed_indices])
    # Calculate transforms with procrustes         
    disparity, transform = procrustes(filtered_reference_coordinates, filtered_submission_coordinates)

    # Convert submission coordinates to world coordinates
    submission_coordinates = transform["scale"] * submission_coordinates @ transform["rotation"] + transform["translation"]
    
    rmses = np.sqrt(((reference_coordinates - submission_coordinates) ** 2).sum(axis=1))
    return rmses


def evaluate_view_synthesis(submission_path, reference_path, crop = False):
    """
    Evaluates view synthesis competition vector, given the submission and reference path.
    First loads all reference and submission images. We then calculate dreamsim on each image pair. 
    May error out if a submission doesn't have all the images present in the reference set.

    :param submission_path: The path to submitted images
    :param reference_path: The path to the reference images
    :raises KeyError: Thrown if an image in the reference set is not present in the submission set (by name)
    :return: The dreamsim score for each image pair in the reference and submission set
    """
    scores = []
    submissions = load_images(submission_path)
    references = load_images(reference_path)
    for name, reference_image in references.items():
        submission_image = submissions.get(name)
        if not submission_image: 
            raise KeyError(f"No such image named {name} in submissions. Please ensure you submitted an image named {name}")
        if crop == False:
            # default is to run on whole image
            scores.append(evaluate_image(reference_image, submission_image))
        else:
            # alternative is to run on crops and average
            # this is intended to increase sensitivity to quality of background features in the image
            h, w = reference_image.size
            box = (0, 0, int(w/2.), int(h/2.))
            score1 = evaluate_image(reference_image.crop(box), submission_image.crop(box))
            box = (0, int(h/2), int(w/2.), h)
            score2 = evaluate_image(reference_image.crop(box), submission_image.crop(box))
            box = (int(w/2.), 0, w, int(h/2.))
            score3 = evaluate_image(reference_image.crop(box), submission_image.crop(box))
            box = (int(w/2.), int(h/2.), w, h)
            score4 = evaluate_image(reference_image.crop(box), submission_image.crop(box))
            box = (int(h/4.), int(w/4.), int(h*3/4.), int(w*3/4.))
            score5 = evaluate_image(reference_image.crop(box), submission_image.crop(box))
            score = (score1 + score2 + score3 + score4 + score5)/5.0
            scores.append(score)
    return scores

if __name__ == "__main__":
    # define input and output paths
    start = time.time()
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    # optionally define other inputs
    try:
        res_label = sys.argv[3]
        crop = sys.argv[4].lower() == "true"
    except:
        res_label = 'res'
        crop = False

    if not os.path.exists(output_path):
        os.makedirs(output_path)
#    submission_path = os.path.join(input_path, 'res')
    submission_path = os.path.join(input_path, res_label)
    reference_path = os.path.join(input_path, 'ref')
    # retrieve full path to camera calibration submission and reference
    calibration_reference_path = os.path.join(reference_path, 'camera_calibration')
    calibration_submission_path = os.path.join(submission_path, 'camera_calibration')
    # list datasets we're using
    calibration_datasets = sorted(os.listdir(calibration_reference_path))

    per_image_results = defaultdict(lambda: defaultdict(dict))
    summary_results = {}

    for dataset in calibration_datasets:
        print(f'Evaluating camera calibration for {dataset}')
        dataset_reference_calibration_path = os.path.join(calibration_reference_path, dataset)
        dataset_submission_calibration_path = os.path.join(calibration_submission_path, dataset)
        dataset_result = evaluate_camera_calibration(dataset_submission_calibration_path, dataset_reference_calibration_path)
        # We save as a "short" name for each dataset (ie, t01, t02...)
        per_image_results['camera_calibration'][dataset] = list(dataset_result)
        short_dataset_name = dataset.split('_')[0]
        summary_results[f'{short_dataset_name}_se90'] = np.percentile(dataset_result, 90)
        summary_results[f'{short_dataset_name}_se50'] = np.percentile(dataset_result, 50)

    view_synth_reference_path = os.path.join(reference_path, 'view_synthesis')
    view_synth_submission_path = os.path.join(submission_path, 'view_synthesis')
    view_synth_datasets = sorted(os.listdir(view_synth_reference_path))

    for dataset in view_synth_datasets:
        print(f'Evaluating view synthesis for {dataset}')
        dataset_reference_view_synth_path = os.path.join(view_synth_reference_path, dataset)
        dataset_submission_view_synth_path = os.path.join(view_synth_submission_path, dataset)
        # If dataset calculation errors for view synthesis due to missing images, we catch and output the max for dsim (1)
        try:
            dataset_result = evaluate_view_synthesis(dataset_submission_view_synth_path, dataset_reference_view_synth_path, crop=crop)
        except Exception:
            print(f"Dataset {dataset} errored out. Score will be max [1.0] for this dataset.")
            print(traceback.format_exc())
            dataset_result = [1.0]

        per_image_results['view_synthesis'][dataset] = dataset_result
        short_dataset_name = dataset.split('_')[0]
        summary_results[f'{short_dataset_name}_dreamsim'] = np.mean(dataset_result)

    with open(os.path.join(output_path, "per_frame_results.json"), "w") as outfile:
        json.dump(per_image_results, outfile)

    # write results to file
    print('Writing scores...')
    fid = open(os.path.join(output_path, 'scores.txt'), "w")
    for key, score in summary_results.items():
        fid.write(f"{key}: {score}\n")
    fid.close()
    end = time.time()
    print(f'Finished. Time to complete: {end - start} seconds.')

