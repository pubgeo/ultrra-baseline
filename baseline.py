"""
TODO ULTRRA CONTEST HEADER
"""

import argparse
import os
import time
from pathlib import Path
import json
import shutil

import numpy as np
from imageio import imwrite
from pymap3d import geodetic2enu
from scipy.spatial.transform import Rotation as R

from baseline_utils.read_write_model import read_model, write_model,replace_imagename_in_model,rename_images,merge_models
from baseline_utils.ultrra_to_colmap import ultrra_to_colmap
from baseline_utils.utils import procrustes, transform_colmap_model

from baseline_utils.nerfstudio_utils import colmap_to_ns_transforms

ROOT_OUT_DATA_DIR = Path("./runs")

DO_FRESH_RUNS_ONLY = False
DELETE_RUN_DIR_WHEN_COMPLETE = False

INTERPOLATION_STEPS = 60


from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
import pycolmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_datasets_dir", type=Path, required=False, help="path to root dir for WACV datasets (should have 'input', 'ref', and 'res' dirs)"
    )
    parser.add_argument(
        '--stage', type=str, required=False, help="stage of the contest to run for ('camera_calibration' or 'view_synthesis')"
    )
    parser.add_argument(
        "--dataset_name", type=Path, required=False, help="name WACV dataset"
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
    # args.stage = Path(f'view_synthesis')
    # args.root_datasets_dir = Path(f'/media/wriva/Data/WACV25/dev_phase/dev_contest_datasets_final/')
    # args.root_datasets_dir = Path(f'/media/wriva/Data/WACV25/test_phase/test_contest_datasets')
    # args.root_datasets_dir = Path(f'/media/wriva/Data/WACV25/dev_phase/dev_contest_datasets_241207/')
    # args.root_datasets_dir = Path(f'/media/wriva/Data/WACV25/test_phase/test_contest_datasets_241207/')
    
    # args.dataset_name = Path(f't02_v06_s00_r01_CameraModels_WACV_dev_A01')
    # args.dataset_name = Path(f't01_v09_s00_r01_ImageDensity_WACV_dev_A01')
    # args.dataset_name = Path(f't03_v06_s00_r01_ReconstructedArea_WACV_dev_A01')
    # args.dataset_name = Path(f't04_v11_s00_r01_VaryingAltitudes_WACV_dev_A01')
    
    # args.dataset_name = Path(f't01_v10_s00_r01_ImageDensity_WACV_test_A09')
    # args.dataset_name = Path(f't02_v07_s00_r01_CameraModels_WACV_test_A09')
    # args.dataset_name = Path(f't03_v07_s00_r01_ReconstructedArea_WACV_test_A09')
    # args.dataset_name = Path(f't04_v12_s00_r01_VaryingAltitudes_WACV_test_A09')
    
    # args.dataset_name = Path(f't04_v12_s00_r01_VaryingAltitudes_WACV_test_A09')
    
    
    # args.dataset_name = Path(f't01_v09_s00_r01_ImageDensity_WACV_dev_A01')
    
    
    hack_cam_location = False
    tmp_image_folder = Path(f'~/ultrra-baseline/images') / args.dataset_name 
    assert args.method_to_use == 'splatfacto', "Still working on testing other nerfstudio methods. Please use splatfacto only for now."

    start_time = time.time()

    # validate/setup dataset dirs
    inputs_dir = args.root_datasets_dir / 'inputs' / args.stage / args.dataset_name
    assert inputs_dir.exists(), f"No inputs dir found at: {inputs_dir}"
    ref_dir = args.root_datasets_dir / 'ref' / args.stage / args.dataset_name
    assert ref_dir.exists(), f"No ref dir found at: {ref_dir}"
    if hack_cam_location:
        res_dir = args.root_datasets_dir / 'res_homer_hack' / args.stage / args.dataset_name
    else:
        res_dir = args.root_datasets_dir / 'res_homer' / args.stage / args.dataset_name
    res_dir.mkdir(exist_ok=True, parents=True)

    
    # setup temporary dir for run, to run COLMAP, nerfstudio, etc. and store intermediate outputs along the pipeline
    run_dir = Path(f"./temp_run_dir_{args.dataset_name}_{args.stage}")
    if DO_FRESH_RUNS_ONLY and run_dir.exists():
        os.rmdir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # run arbitrary colmap
    train_images_dir = inputs_dir if str(args.stage) == 'camera_calibration' else inputs_dir / 'train'
    train_images_dir = inputs_dir if str(args.stage) == 'camera_calibration' else inputs_dir / 'train'
    arb_colmap_dir = run_dir / 'arb_colmap'
    img_cource = arb_colmap_dir / 'images'
    
    if not arb_colmap_dir.exists():
        outputs = arb_colmap_dir / Path("colmap/")
        sfm_pairs = outputs / "pairs-exhaustive.txt"
        sfm_dir = outputs / "sparse" /"0"
        feature_conf = extract_features.confs["superpoint_max"]
        matcher_conf = match_features.confs["superglue"]

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
            match_path,verbose=True,camera_mode='PER_IMAGE',image_options = image_options,mapper_options=mapper_options,
            min_match_score = 0.1,skip_geometric_verification=False)
        cameras, images, points3D = read_model(arb_colmap_dir / "colmap" / "sparse" / "0", ext=".bin")
        write_model(cameras, images, points3D, arb_colmap_dir / "colmap" / "sparse" / "0", ext=".txt")
        if not img_cource.exists():
            os.makedirs(img_cource)
        
        for basename in os.listdir(train_images_dir):
            if basename.endswith('.jpg'):
                pathname = train_images_dir / basename
                if os.path.isfile(pathname):
                    shutil.copy2(pathname, img_cource)

        
    # if not arb_colmap_dir.exists():
    #     # os.system(f"ns-process-data images --data {train_images_dir} --output-dir {arb_colmap_dir}  --matching-method {args.colmap_matching_method} \
    #     #     --skip_image_processing --sfm-tool hloc --feature-type superpoint_aachen --matcher-type superglue\
    #     #     --no-same-dimensions --no-use-single-camera-mode")
    #     os.system(f"ns-process-data images --data {train_images_dir} --output-dir {arb_colmap_dir}  --matching-method {args.colmap_matching_method} \
    #         --sfm-tool hloc --feature-type superpoint_max --matcher-type superglue  --use-sfm-depth --refine-intrinsics --camera-type perspective\
    #         --no-same-dimensions --no-use-single-camera-mode")
        
    # one of the prob with nerfstudio is the renaming process during copying images. However, we still want to keep the original image names
    # the solution is to figure out the mapping between the original and the new names
    print('create image mapping')

    # the original name list
    allowed_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] 
    glob_str =  "[!.]*"
    org_files = sorted([os.path.basename(p) for p in train_images_dir.glob(glob_str) if p.suffix.lower() in allowed_exts])
    new_files = sorted([os.path.basename(p) for p in img_cource.glob('frame*.*')])
    
    arb_model_root = arb_colmap_dir / "colmap" / "sparse" / "0" / "models"  
    # this part is not required anymore, leave it here just in case
    if len(org_files)==len(new_files): # we do mapping only the first time when frame_* exists
        mapping = {}
        for new_file,org_file in zip(new_files,org_files):
            mapping[new_file] = org_file
        
        for dir in arb_model_root.iterdir():
            if dir.is_dir():
                replace_imagename_in_model(dir,mapping)
        replace_imagename_in_model(arb_colmap_dir / "colmap" / "sparse" / "0",mapping)
        rename_images(arb_colmap_dir,mapping)
    
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
        os.chdir(run_dir / "combined")

        # train, render, and evaluate
        if 'splatfacto' in args.method_to_use:
            os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ns-train {args.method_to_use} --method-name {args.method_to_use} --data . --vis viewer+tensorboard --viewer.quit-on-train-completion True nerfstudio-data")
        else:
            # to support variable image resolutions, we have to use a special datamanager for non-splatfacto methods
            os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ns-train {args.method_to_use} --method-name {args.method_to_use} --data . --vis viewer+tensorboard --viewer.quit-on-train-completion True --pipeline.datamanager.train-num-images-to-sample-from 40 --pipeline.datamanager.train-num-times-to-repeat-images 100 --pipeline.datamanager.eval-num-images-to-sample-from 40 --pipeline.datamanager.eval-num-times-to-repeat-images 100 {'--pipeline.model.predict-normals True' if 'nerfacto' in args.method_to_use else ''} nerfstudio-data")
        # os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ns-train {args.method_to_use} --method-name {args.method_to_use} --data . --viewer.quit-on-train-completion True nerfstudio-data")

        config_path = list(Path(f"./outputs/{args.method_to_use}").glob("*/config.yml"))
        #assert len(config_path) == 1
        config_path = config_path[0]

        # render dataset outputs
        os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ns-render dataset --load-config {config_path} --data . --split train+test --rendered-output-names rgb gt-rgb depth")

        # finally we can move rendered results to /res folder, completing the view_synthesis stage
        os.system(f"cp ./renders/test/rgb/* {res_dir}/")

        # bonus: render interpolated trajectory between ref views
        os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ns-render interpolate --load-config {config_path} --interpolation-steps {INTERPOLATION_STEPS} --output-path ./renders/eval_interp_raw.mp4 --downscale-factor 2")

    if DELETE_RUN_DIR_WHEN_COMPLETE:
        os.chdir(run_dir.parent)
        os.rmdir(run_dir)

    print(f"Finished run for dataset, {args.dataset_name} ({args.stage} stage), in {((time.time() - start_time) / 60.0 / 60.0):.2f} hours")


if __name__ == "__main__":
    main()
