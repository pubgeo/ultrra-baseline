""" Convert WRIVA datasets to ULTRRA format.
"""

from pathlib import Path
from glob import glob
import os
import json
import math
import shutil
from tqdm import tqdm
import csv
import cv2
from PIL import Image

def convert_wrivacraft_to_ultrra(input_path, output_path):
    """
    convert full WrivaCraft challenge datasets to ULTRRA format
    :param input_path: input path containing WrivaCraft datasets
    :param output_path: output path for writing datasets formatted to match the contest
    """
    os.makedirs(output_path, exist_ok=True)
    dataset_paths = glob(os.path.join(input_path,'*'))
    for dataset_path in dataset_paths:
        if not os.path.isdir(dataset_path): continue
        dataset_basename = os.path.basename(dataset_path)
        # copy reference data
        image_dir = os.path.join(dataset_path, 'reference', 'images')
        json_dir = os.path.join(dataset_path, 'reference', 'metadata')
        json_paths = sorted(glob(os.path.join(json_dir,'*.json')))
        print(dataset_basename)
        for json_path in tqdm(json_paths):
            # load JSON metadata
            with open(json_path) as f:
                params = json.load(f)
            # load image
            input_image_file = os.path.join(image_dir, params['fname'])
            img = cv2.imread(input_image_file)
            # save reference image for view synthesis reference
            output_image_dir = os.path.join(output_path, 'ref', 'view_synthesis')
            output_image_dir = os.path.join(output_image_dir, dataset_basename)
            os.makedirs(output_image_dir, exist_ok=True)
            output_json_name = os.path.basename(json_path)
            output_image_name = output_json_name.replace('.json', '.jpg')
            output_image_file = os.path.join(output_image_dir, output_image_name)
            cv2.imwrite(output_image_file, img)
            # save reference metadata for view synthesis test input
            params_metadata = {}
            params_metadata['fname'] = output_image_name
            params_metadata['timestamp'] = params['timestamp']
            params_metadata['type'] = params['type']
            params_metadata['geolocation'] = params['geolocation']
            params_metadata['intrinsics'] = params['intrinsics']
            params_metadata['extrinsics'] = params['extrinsics']
            reference_metadata_dir = os.path.join(output_path, 'inputs', 'view_synthesis')
            reference_metadata_dir = os.path.join(reference_metadata_dir, dataset_basename, 'test')
            os.makedirs(reference_metadata_dir, exist_ok=True)
            output_json_file = os.path.join(reference_metadata_dir, output_json_name)
            with open(output_json_file, 'w') as f:
                json.dump(params_metadata, f, indent=2)
        # copy input data
        image_dir = os.path.join(dataset_path, 'input', 'images')
        json_dir = os.path.join(dataset_path, 'reference', 'ta2_metadata')
        json_paths = sorted(glob(os.path.join(json_dir,'*.json')))
        print(dataset_basename)
        for json_path in tqdm(json_paths):
            # load JSON metadata
            with open(json_path) as f:
                params = json.load(f)
            # load image
            extension = os.path.splitext(params['fname'])[1]            
            input_image_file = os.path.join(image_dir, os.path.basename(json_path).replace('.json', extension))
            img = cv2.imread(input_image_file)
            # save input image for view synthesis train input
            output_image_dir = os.path.join(output_path, 'inputs', 'view_synthesis')
            output_image_dir = os.path.join(output_image_dir, dataset_basename, 'train')
            os.makedirs(output_image_dir, exist_ok=True)
            output_json_name = os.path.basename(json_path)
            output_image_name = output_json_name.replace('.json', '.jpg')
            output_image_file = os.path.join(output_image_dir, output_image_name)
            cv2.imwrite(output_image_file, img)
            # save JSONs for view synthesis train input
            params_metadata = {}
            params_metadata['fname'] = output_image_name
            params_metadata['extrinsics'] = {}
            params_metadata['extrinsics']['lat'] = params['extrinsics']['lat']
            params_metadata['extrinsics']['lon'] = params['extrinsics']['lon']
            params_metadata['extrinsics']['alt'] = params['extrinsics']['alt']
            params_metadata['timestamp'] = params['timestamp']
            params_metadata['type'] = params['type']
            params_metadata['geolocation'] = params['geolocation']
            input_metadata_dir = os.path.join(output_path, 'inputs', 'view_synthesis')
            input_metadata_dir = os.path.join(input_metadata_dir, dataset_basename, 'train')
            os.makedirs(input_metadata_dir, exist_ok=True)
            output_json_file = os.path.join(input_metadata_dir, output_json_name)
            with open(output_json_file, 'w') as f:
                json.dump(params_metadata, f, indent=2)
            # save input image for camera calibration input
            output_image_dir = os.path.join(output_path, 'inputs', 'camera_calibration')
            output_image_dir = os.path.join(output_image_dir, dataset_basename)
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_file = os.path.join(output_image_dir, output_image_name)
            cv2.imwrite(output_image_file, img)
            # save full input metadata for camera calibration reference
            params_metadata['intrinsics'] = params['intrinsics']
            params_metadata['extrinsics'] = params['extrinsics']
            reference_metadata_dir = os.path.join(output_path, 'ref', 'camera_calibration')
            reference_metadata_dir = os.path.join(reference_metadata_dir, dataset_basename)
            os.makedirs(reference_metadata_dir, exist_ok=True)
            output_json_file = os.path.join(reference_metadata_dir, output_json_name)
            with open(output_json_file, 'w') as f:
                json.dump(params_metadata, f, indent=2)

if __name__ == "__main__":
    # Convert DataPort WrivaCraft datasets to ULTRRA format
    # Replace path names with yours after downloading them from DataPort
    input_path = str(Path(r'Y:\Data\DATAPORT\wriva-challenge-datasets'))
    output_path = str(Path(r'Y:\Data\DATAPORT\wriva-challenge-datasets-ultrra'))
    convert_wrivacraft_to_ultrra(input_path, output_path)


