"""
TODO ULTRRA CONTEST HEADER
"""

import argparse
import os
import time
from pathlib import Path
import json

import numpy as np
from imageio import imwrite
from pymap3d import geodetic2enu
from scipy.spatial.transform import Rotation as R

from baseline_utils.read_write_model import read_model, write_model
from baseline_utils.ultrra_to_colmap import ultrra_to_colmap
from baseline_utils.utils import procrustes, transform_colmap_model

from baseline_utils.nerfstudio_utils import colmap_to_ns_transforms

ROOT_OUT_DATA_DIR = Path("./runs")

DO_FRESH_RUNS_ONLY = False
DELETE_RUN_DIR_WHEN_COMPLETE = False

INTERPOLATION_STEPS = 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_datasets_dir", type=Path, required=True, help="path to root dir for WACV datasets (should have 'input', 'ref', and 'res' dirs)"
    )
    parser.add_argument(
        '--stage', type=str, required=True, help="stage of the contest to run for ('camera_calibration' or 'view_synthesis')"
    )
    parser.add_argument(
        "--dataset_name", type=Path, required=True, help="name WACV dataset"
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
    assert args.method_to_use == 'splatfacto', "Still working on testing other nerfstudio methods. Please use splatfacto only for now."

    start_time = time.time()

    # validate/setup dataset dirs
    inputs_dir = args.root_datasets_dir / 'inputs' / args.stage / args.dataset_name
    assert inputs_dir.exists(), f"No inputs dir found at: {inputs_dir}"
    ref_dir = args.root_datasets_dir / 'ref' / args.stage / args.dataset_name
    assert ref_dir.exists(), f"No ref dir found at: {ref_dir}"
    res_dir = args.root_datasets_dir / 'res' / args.stage / args.dataset_name
    res_dir.mkdir(exist_ok=True, parents=True)
    
    # setup temporary dir for run, to run COLMAP, nerfstudio, etc. and store intermediate outputs along the pipeline
    run_dir = Path(f"./temp_run_dir_{args.dataset_name}_{args.stage}")
    if DO_FRESH_RUNS_ONLY and run_dir.exists():
        os.rmdir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # run arbitrary colmap
    train_images_dir = inputs_dir if args.stage == 'camera_calibration' else inputs_dir / 'train'
    arb_colmap_dir = run_dir / 'arb_colmap'
    if not arb_colmap_dir.exists():
        os.system(f"ns-process-data images --data {train_images_dir} --output-dir {arb_colmap_dir} --num_downscales 0 --matching-method {args.colmap_matching_method}")
    arb_colmap_model = read_model(arb_colmap_dir / "colmap" / "sparse" / "0", ext=".bin")
    arb_colmap_name_id_map = {colmap_image_obj.name: id for id, colmap_image_obj in arb_colmap_model[1].items()}

    # calculate colmap cartesian coords (x, y, z) for training data
    successful_arb_colmap_cart_dict = {}
    for im_path in sorted([path for path in train_images_dir.glob("*") if path.suffix != '.json']):
        try:
            colmap_image_obj = arb_colmap_model[1][arb_colmap_name_id_map[f"frame_{im_path.stem}.jpg"]]
            x, y, z = -R.from_quat(np.roll(colmap_image_obj.qvec, -1)).inv().apply(colmap_image_obj.tvec)

            predicted_data = {'fname': im_path.name, 'x': x, 'y': y, 'z': z, 'success': True}
            successful_arb_colmap_cart_dict[im_path.stem] = [x, y, z]
        except Exception as e:
            # print(e)
            predicted_data = {'fname': im_path.name, 'x': 0, 'y': 0, 'z': 0, 'success': False}

        # for camera_calibration stage, we output .jsons with these x, y, z values as our contest output in /res, then we're done
        if args.stage == 'camera_calibration':
            json.dump(predicted_data, open(res_dir / f"{im_path.stem}.json", 'w'), indent=2)

    if args.stage == 'view_synthesis':
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
        for im_name in sorted(successful_arb_colmap_cart_dict.keys()):
            arb_colmap_enu_list.append(successful_arb_colmap_cart_dict[im_name])
            
            # convert provided train lat/lon/alt to enu
            provided_input_metadata = json.load(open(inputs_dir / 'train' / f"{im_name}.json"))
            provided_enu_list.append(geodetic2enu(provided_input_metadata['extrinsics']['lat'], provided_input_metadata['extrinsics']['lon'], provided_input_metadata['extrinsics']['alt'], origin[0], origin[1], origin[2]))        # TODO do I need to swap origin[0] and origin[1] here because lat/lon is y/x ???
        
        # from validly calibrated images, run procrustes to get transformation from arbitrary colmap to provided input space
        disparity, transform = procrustes(provided_enu_list, arb_colmap_enu_list)

        # run transform_colmap_model to transform arbitrary colmap model to provided input space, using above transform
        transformed_arb_colmap_model = transform_colmap_model(arb_colmap_model, transform)
        transformed_arb_colmap_dir = run_dir / "transformed_arb_colmap"
        transformed_arb_colmap_sparse_0_dir = transformed_arb_colmap_dir / "sparse" / "0"
        transformed_arb_colmap_sparse_0_dir.mkdir(exist_ok=True, parents=True)
        write_model(cameras=transformed_arb_colmap_model[0], images=transformed_arb_colmap_model[1], points3D=transformed_arb_colmap_model[2], path=transformed_arb_colmap_sparse_0_dir, ext=".txt")
        (transformed_arb_colmap_dir / 'colmap').mkdir(exist_ok=True, parents=True)
        os.system(f"mv {transformed_arb_colmap_dir / 'sparse'} {transformed_arb_colmap_dir / 'colmap/'}")
        os.system(f"cp -r {arb_colmap_dir / 'images'} {transformed_arb_colmap_dir / 'images'}")
        
        # create nerfstudio-style transforms.json for transformed arbitrary colmap model
        colmap_to_ns_transforms(transformed_arb_colmap_model, transformed_arb_colmap_dir / 'transforms.json')

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
        assert len(config_path) == 1
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
