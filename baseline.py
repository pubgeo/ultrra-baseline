"""
TODO ULTRRA CONTEST HEADER
"""

import argparse
import os
import time
from pathlib import Path
import json
import shutil
from glob import glob

import numpy as np
from imageio import imwrite
from pymap3d import geodetic2enu
from scipy.spatial.transform import Rotation as R

from baseline_utils.read_write_model import read_model, write_model,replace_imagename_in_model,rename_images,merge_models
from baseline_utils.ultrra_to_colmap import ultrra_to_colmap
from baseline_utils.utils import procrustes, transform_colmap_model

from baseline_utils.nerfstudio_utils import colmap_to_ns_transforms

INTERPOLATION_STEPS = 60

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)

def baseline(args):
    start_time = time.time()

    # validate/setup dataset dirs
    inputs_dir = args.root_datasets_dir / 'inputs' / args.stage / args.dataset_name
    assert inputs_dir.exists(), f"No inputs dir found at: {inputs_dir}"

    res_dir = args.root_datasets_dir / args.output_name / args.stage / args.dataset_name
    res_dir.mkdir(exist_ok=True, parents=True)
    
    # setup dir for run, to run COLMAP, nerfstudio, etc. and store intermediate outputs along the pipeline
    # if previously ran but no valid output, then delete old run directory and try again
    # else this is done so skip
    run_dir = args.root_datasets_dir / args.output_name / "run" / args.stage / args.dataset_name
    if run_dir.exists():
        if len(run_dir.glob("*")) == 0:
            shutil.rmtree(run_dir)
        else:
            print(f"Run directory already exists so skipping: {run_dir}")
            return
    run_dir.mkdir(parents=True, exist_ok=True)

    # run arbitrary colmap
    train_images_dir = inputs_dir if str(args.stage) == 'camera_calibration' else inputs_dir / 'train'
    arb_colmap_dir = run_dir / 'arb_colmap'
    img_source = arb_colmap_dir / 'images'
    
    if not arb_colmap_dir.exists():
        outputs = arb_colmap_dir / Path("colmap/")
        sfm_pairs = outputs / "pairs-exhaustive.txt"
        sfm_dir = outputs / "sparse" /"0"
#        feature_conf = extract_features.confs["superpoint_max"]
        feature_conf = extract_features.confs["superpoint_aachen"]
#        matcher_conf = match_features.confs["superglue"]
        matcher_conf = match_features.confs["superpoint+lightglue"]

        # image_options={'camera_model':'FULL_OPENCV'}
        image_options={}
        # https://github.com/colmap/colmap/blob/main/src/colmap/controllers/incremental_pipeline.h
        # mapper_options = {"min_focal_length_ratio": 0.1, "max_focal_length_ratio": 10}
        mapper_options = {}
        feature_path = extract_features.main(feature_conf, train_images_dir, outputs)
        pairs_from_exhaustive.main(sfm_pairs,features=feature_path)
        match_path = match_features.main(
            matcher_conf, sfm_pairs, feature_conf["output"], outputs
        )
        model = reconstruction.main(sfm_dir, train_images_dir, sfm_pairs, feature_path, 
            match_path, verbose=False, camera_mode='PER_IMAGE', image_options=image_options, mapper_options=mapper_options,
            min_match_score = 0.1, skip_geometric_verification=False)
        cameras, images, points3D = read_model(arb_colmap_dir / "colmap" / "sparse" / "0", ext=".bin")
        write_model(cameras, images, points3D, arb_colmap_dir / "colmap" / "sparse" / "0", ext=".txt")
        if not img_source.exists():
            os.makedirs(img_source)
        
        for basename in os.listdir(train_images_dir):
            if basename.endswith('.jpg'):
                pathname = train_images_dir / basename
                if os.path.isfile(pathname):
                    shutil.copy2(pathname, img_source)

    arb_model_root = arb_colmap_dir / "colmap" / "sparse" / "0" / "models"  
    
    #read multiple models    
    arb_colmap_models = []
    arb_colmap_name_id_maps = []
    
    cur_model = read_model(arb_colmap_dir / "colmap" / "sparse" / "0", ext=".bin")
    arb_colmap_name_id_maps.append({colmap_image_obj.name: id for id, colmap_image_obj in cur_model[1].items()})
    arb_colmap_models.append(cur_model)

    for dir in arb_model_root.iterdir():
        if dir.is_dir():
            try:
                cur_model = read_model(dir, ext=".bin")
                arb_colmap_name_id_maps.append({colmap_image_obj.name: id for id, colmap_image_obj in cur_model[1].items()})
                arb_colmap_models.append(cur_model)
            except:
                continue
    
    # calculate colmap cartesian coords (x, y, z) for training data
    successful_arb_colmap_cart_dicts = []
    for i in range(len(arb_colmap_models)):
        successful_arb_colmap_cart_dicts.append({})

    for im_path in sorted([path for path in train_images_dir.glob("*") if path.suffix != '.json']):
        foundKey = False
        for idx,(arb_colmap_name_id_map, arb_colmap_model) in enumerate(zip(arb_colmap_name_id_maps, arb_colmap_models)):
            cur_key = f"{im_path.stem}.jpg"
            if cur_key in arb_colmap_name_id_map.keys():
                colmap_image_obj = arb_colmap_model[1][arb_colmap_name_id_map[cur_key]]
                assert colmap_image_obj.name==cur_key
                #https://colmap.github.io/format.html
                x, y, z = -R.from_quat(np.roll(colmap_image_obj.qvec, -1)).inv().apply(colmap_image_obj.tvec)
                # q = colmap_image_obj.qvec
                # x, y, z = -R.from_quat([q[1],q[2],q[3],q[0]]).inv().apply(colmap_image_obj.tvec)
                if cur_key in arb_colmap_name_id_maps[0].keys():
                    predicted_data = {'fname': im_path.name, 'x': x, 'y': y, 'z': z, 'success': True}
                else:
                    predicted_data = {'fname': im_path.name, 'x': 0, 'y': 0, 'z': 0, 'success': False}
                successful_arb_colmap_cart_dicts[idx][im_path.stem] = [x, y, z]
                foundKey = True
        if not foundKey:
            predicted_data = {'fname': im_path.name, 'x': 0, 'y': 0, 'z': 0, 'success': False}

        # if input JSON metadata indicates inaccurate camera locations, then do not use for procrustes fit
        # this is not an issue for ULTRRA Challenge datasets
        # it is an issue for some of the WRIVA challenge datasets for anyone converting those to run with ULTRRA solutions and metrics
        if str(args.stage) == 'view_synthesis':
            json_path = str(im_path).replace('.jpg', '.json')
            params = json.load(open(json_path))
            if "geolocation" in params:
                if params["geolocation"] != "gcp" and params["geolocation"] != "rtk" and params["geolocation"] != "synthetic":
                    predicted_data['success'] = False

        # for camera_calibration stage, we output .jsons with these x, y, z values as our contest output in /res, then we're done
        if str(args.stage) == 'camera_calibration':
            json.dump(predicted_data, open(res_dir / f"{im_path.stem}.json", 'w'), indent=2)

    disparities = []
    if str(args.stage) == 'view_synthesis':
        # convert test poses to colmap
        test_colmap_dir = run_dir / 'test_colmap'
        (test_colmap_dir / 'images').mkdir(parents=True, exist_ok=True)
        # os.system(f"cp {inputs_dir}/test/*.json {test_colmap_dir / 'images'}/")
        os.system(f"cp {inputs_dir}/test/*.json {test_colmap_dir / 'images'}/")
        ultrra_to_colmap(test_colmap_dir, camera_model="OPENCV", save_ext=".txt")
        (test_colmap_dir / 'colmap').mkdir(exist_ok=True, parents=True)
        os.system(f"mv {test_colmap_dir / 'sparse'} {test_colmap_dir / 'colmap/'}")
        test_colmap_model = read_model(test_colmap_dir / "colmap" / "sparse" / "0", ext=".txt")
        
        # create nerfstudio-style transforms.json for test colmap model
        colmap_to_ns_transforms(test_colmap_model, test_colmap_dir / 'transforms.json')
        
        origin_split = open(test_colmap_dir / 'colmap' / 'sparse' / '0' / 'origin.txt').read().replace(",", "").split()
        origin = float(origin_split[1]), float(origin_split[3]), float(origin_split[5])

        # also need to write dummy black images to test_colmap dir
        for json_path in sorted(list((test_colmap_dir / 'images').glob("*.json"))):
            data = json.load(open(json_path))
            imwrite(json_path.with_suffix('.jpg'), np.zeros((data['intrinsics']['rows'], data['intrinsics']['columns'], 3)).astype(np.uint8))

        arb_colmap_enu_list = []
        provided_enu_list = []
        transforms = []
        transformed_arb_colmap_models = []
        for successful_arb_colmap_cart_dict,arb_colmap_model in zip(successful_arb_colmap_cart_dicts,arb_colmap_models):
            for im_name in sorted(successful_arb_colmap_cart_dict.keys()):
                arb_colmap_enu_list.append(successful_arb_colmap_cart_dict[im_name])
                
                # convert provided train lat/lon/alt to enu
                provided_input_metadata = json.load(open(inputs_dir / 'train' / f"{im_name}.json"))
                provided_enu_list.append(geodetic2enu(provided_input_metadata['extrinsics']['lat'], provided_input_metadata['extrinsics']['lon'], provided_input_metadata['extrinsics']['alt'], origin[0], origin[1], origin[2]))        # TODO do I need to swap origin[0] and origin[1] here because lat/lon is y/x ???
            
            # from validly calibrated images, run procrustes to get transformation from arbitrary colmap to provided input space
            disparity, transform = procrustes(provided_enu_list, arb_colmap_enu_list)
            disparities.append(disparity)
            transforms.append(transform)
            transformed_arb_colmap_models.append(transform_colmap_model(arb_colmap_model, transform))
        merged_models = merge_models(transformed_arb_colmap_models,disparities)    
        #merge models
        # run transform_colmap_model to transform arbitrary colmap model to provided input space, using above transform
        # transformed_arb_colmap_model = transform_colmap_model(arb_colmap_model, transform)
        transformed_arb_colmap_dir = run_dir / "transformed_arb_colmap"
        transformed_arb_colmap_sparse_0_dir = transformed_arb_colmap_dir / "sparse" / "0"
        transformed_arb_colmap_sparse_0_dir.mkdir(exist_ok=True, parents=True)
        write_model(cameras=merged_models[0], images=merged_models[1], points3D=merged_models[2], path=transformed_arb_colmap_sparse_0_dir, ext=".txt")
        (transformed_arb_colmap_dir / 'colmap').mkdir(exist_ok=True, parents=True)
        os.system(f"mv {transformed_arb_colmap_dir / 'sparse'} {transformed_arb_colmap_dir / 'colmap/'}")
        os.system(f"cp -r {arb_colmap_dir / 'images'} {transformed_arb_colmap_dir / 'images'}")
        
        # create nerfstudio-style transforms.json for transformed arbitrary colmap model
        colmap_to_ns_transforms(merged_models, transformed_arb_colmap_dir / 'transforms.json')

        # combine test data and transforms.json and transformed arbitrary colmap data and transforms.json
        combined_dir = run_dir / "combined"
        (combined_dir / "images").mkdir(exist_ok=True, parents=True)
        os.system(f"cp {test_colmap_dir / 'images'}/* {combined_dir / 'images'}/")
        os.system(f"cp {arb_colmap_dir / 'images'}/* {combined_dir / 'images'}/")

        test_colmap_dir_transforms = json.load(open(test_colmap_dir / "transforms.json", 'r'))
        transformed_arb_colmap_dir_transforms = json.load(open(transformed_arb_colmap_dir / "transforms.json", 'r'))
        test_filenames = [frame['file_path'] for frame in test_colmap_dir_transforms['frames']]
        train_filenames = [frame['file_path'] for frame in transformed_arb_colmap_dir_transforms['frames']]
        
        combined_transforms = test_colmap_dir_transforms.copy()
        combined_transforms['frames'] = combined_transforms['frames'] + transformed_arb_colmap_dir_transforms['frames']
        combined_transforms['train_filenames'] = train_filenames
        combined_transforms['val_filenames'] = test_filenames
        combined_transforms['test_filenames'] = test_filenames

        json.dump(combined_transforms, open(combined_dir / "transforms.json", 'w'), indent=4)

        # data preprocessing is done, now we can run nerfstudio
        # change the path so expected files are at the top level
        current_path = os.getcwd()
        os.chdir(run_dir / "combined")

        # train, render, and evaluate
        if 'splatfacto' in args.method_to_use:
            os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ns-train {args.method_to_use} --method-name {args.method_to_use} --data . --vis viewer+tensorboard --viewer.quit-on-train-completion True nerfstudio-data")
        else:
            # to support variable image resolutions, we have to use a special datamanager for non-splatfacto methods
            #os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ns-train {args.method_to_use} --method-name {args.method_to_use} --data . --vis viewer+tensorboard --viewer.quit-on-train-completion True --pipeline.datamanager.train-num-images-to-sample-from 40 --pipeline.datamanager.train-num-times-to-repeat-images 100 --pipeline.datamanager.eval-num-images-to-sample-from 40 --pipeline.datamanager.eval-num-times-to-repeat-images 100 {'--pipeline.model.predict-normals True' if 'nerfacto' in args.method_to_use else ''} nerfstudio-data")
            os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ns-train {args.method_to_use} --method-name {args.method_to_use} --data . --viewer.quit-on-train-completion True nerfstudio-data")

        config_path = list(Path(f"./outputs/{args.method_to_use}").glob("*/config.yml"))
        config_path = config_path[0]

        # render dataset outputs
        os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ns-render dataset --load-config {config_path} --data . --split test --rendered-output-names rgb depth")

        # finally we can move rendered results to /res folder, completing the view_synthesis stage
        os.system(f"cp ./renders/test/rgb/* {res_dir}/")
        depth_dir = os.path.join(res_dir, 'depth')
        os.makedirs(depth_dir, exist_ok=True)
        os.system(f"cp ./renders/test/depth/* {depth_dir}")

        # bonus: render interpolated trajectory between ref views
        try:
            os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ns-render interpolate --load-config {config_path} --interpolation-steps {INTERPOLATION_STEPS} --output-path ./renders/eval_interp_raw.mp4 --downscale-factor 2")
        except:
            print("Error: Failed to produce flythrough. Check that FFMPEG is installed.")

        # change path back to what it was before running nerfstudio
        os.chdir(current_path)

    print(f"Finished run for dataset, {args.dataset_name} ({args.stage} stage), in {((time.time() - start_time) / 60.0 / 60.0):.2f} hours")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_datasets_dir", type=Path, required=True, help="path to root dir for datasets (should include 'input', outputs)"
    )
    parser.add_argument(
        '--stage', type=str, required=False, default="camera_calibration", help="stage of the contest to run for ('camera_calibration' or 'view_synthesis')"
    )
    parser.add_argument(
        "--dataset_name", type=Path, required=False, default=None, help="input dataset name"
    )
    parser.add_argument(
        "--output_name", type=str, default="res", required=False, help="default is 'res'"
    )
    parser.add_argument(
        "--cuda_visible_devices", type=str, required=False, default="0", help="device number of GPU to use for nerfstudio training and rendering"
    )
    parser.add_argument(
        '--colmap_matching_method', type=str, required=False, default="exhaustive", help="type of feature matching method for colmap ('vocab_tree' or 'exhaustive')"
    )
    parser.add_argument(
        "--method_to_use", type=str, required=False, default="splatfacto", help="which nerfacto model to use for 'view_synthesis' stage (ex: 'nerfacto', 'splatfacto', etc)"
    )
    args = parser.parse_args()
    if args.dataset_name == None:
        search_path = os.path.join(args.root_datasets_dir, 'inputs', 'camera_calibration')
        dataset_paths = sorted(glob(os.path.join(search_path,'*')))
        for dataset_path in dataset_paths:
            args.dataset_name = os.path.basename(dataset_path)
            baseline(args)
    else:
        baseline(args)

if __name__ == "__main__":
    main()
