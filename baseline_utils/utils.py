"""
TODO ULTRRA CONTEST HEADER
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pymap3d as pm
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation as R

from .read_write_model import (
    Camera,
    Image,
    Point3D,
    read_model,
    write_model,
)

logging.captureWarnings(True)
logger = logging.getLogger(__name__)


def metadata_to_model(metadata_dicts, origin, camera_model="FULL_OPENCV"):
    """
    Converts ULTRRA's list of metadata json dictionaries to COLMAP's dictionary objects.

    :param metadata_dicts: list of metadata json dictionaries
    :param origin: origin (lat, lon, alt) of ENU coordinate system
    :return: COLMAP dictionary objects (cameras, images, points3D)
    """
    # create COLMAP cameras
    cams = sorted(
        list(
            set(
                [
                    json.dumps(metadata_dict["intrinsics"])
                    for metadata_dict in metadata_dicts
                ]
            )
        )
    )
    cameras = {}
    if camera_model == 'FULL_OPENCV':
        for i, cam in enumerate(cams):
            d = json.loads(cam)
            params = [
                d[key] for key in ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
            ]
            params.extend([0, 0, 0])
            cameras[i + 1] = Camera(
                id=i + 1,
                model="FULL_OPENCV",
                width=int(d["columns"]),
                height=int(d["rows"]),
                params=np.array(params),
            )
    elif camera_model == "OPENCV":
        for i, cam in enumerate(cams):
            d = json.loads(cam)
            params = [
                d[key] for key in ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
            ]
            params.extend([0, 0, 0])
            cameras[i + 1] = Camera(
                id=i + 1,
                model="OPENCV",
                width=int(d["columns"]),
                height=int(d["rows"]),
                params=np.array(params),
            )

    # create COLMAP images
    images = {}
    for i, metadata_dict in enumerate(metadata_dicts):
        d = metadata_dict["extrinsics"]
        r = (
            R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            * R.from_euler(
                "zyx",
                [d["kappa"], d["phi"], d["omega"]],
                degrees=True,
            ).inv()
        )
        images[i + 1] = Image(
            id=i + 1,
            qvec=np.roll(r.as_quat(), 1),
            tvec=r.apply(
                -np.array(pm.geodetic2enu(d["lat"], d["lon"], d["alt"], *origin))
            ),
            camera_id=cams.index(json.dumps(metadata_dict["intrinsics"])) + 1,
            name=metadata_dict["fname"],
            xys=np.empty([0, 2]),
            point3D_ids=np.empty(0),
        )

    # create COLMAP points3D
    points3D = {}

    return cameras, images, points3D


def model_to_metadata(model, origin, m_dicts):
    """
    Converts COLMAP's dictionary objects to ULTRRA's list of metadata json dictionaries.

    :param model: COLMAP dictionary objects (cameras, images, points3D) of model
    :param origin: origin (lat, lon, alt) of ENU coordinate system
    :param m_dicts: list of metadata json dictionaries
    :return: list of metadata json dictionaries
    """
    # extract objects
    cameras, images, _ = model

    # create list of metadata json dictionaries
    keys_i = ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "rows", "columns"]
    keys_e = ["lat", "lon", "alt", "omega", "phi", "kappa"]
    metadata_dicts = []
    for image in images.values():
        # initialize dictionary
        metadata_dict = {"fname": image.name}
        for m_dict in m_dicts:
            if m_dict["fname"] == metadata_dict["fname"]:
                metadata_dict = m_dict
                break
        metadata_dict["extrinsics"] = dict.fromkeys(keys_e, 0.0)
        metadata_dict["intrinsics"] = dict.fromkeys(keys_i, 0.0)

        # set extrinsics parameters
        r = R.from_quat(np.roll(image.qvec, -1))
        t = -r.inv().apply(image.tvec)
        (
            metadata_dict["extrinsics"]["lat"],
            metadata_dict["extrinsics"]["lon"],
            metadata_dict["extrinsics"]["alt"],
        ) = pm.enu2geodetic(*t, *origin)
        (
            metadata_dict["extrinsics"]["kappa"],
            metadata_dict["extrinsics"]["phi"],
            metadata_dict["extrinsics"]["omega"],
        ) = (
            (R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) * r)
            .inv()
            .as_euler("zyx", degrees=True)
        )

        # set intrinsics parameters
        camera = cameras[image.camera_id]
        if camera.model == "PINHOLE":
            keys_m = ["fx", "fy", "cx", "cy"]
            params = camera.params
        elif camera.model == "SIMPLE_RADIAL":
            keys_m = ["fx", "fy", "cx", "cy", "k1"]
            params = np.insert(camera.params, 0, camera.params[0])
        elif camera.model == "OPENCV":
            keys_m = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
            params = camera.params
        else:
            raise ValueError("{} camera model is not supported".format(camera.model))
        assert len(keys_m) == len(params)
        for key, param in zip(keys_m, params):
            metadata_dict["intrinsics"][key] = param
        metadata_dict["intrinsics"]["rows"] = camera.height
        metadata_dict["intrinsics"]["columns"] = camera.width

        # create list of dictionaries
        metadata_dicts.append(metadata_dict)

    return metadata_dicts


def read_metadata(images_dir):
    """
    Read ULTRRA's image json files.

    :param images_dir: path to images directory
    :return: list of metadata json dictionaries
    """
    metadata_dicts = []
    json_paths = sorted(list(Path(images_dir).rglob("*.json")))
    for json_path in json_paths:
        with open(json_path) as fid:
            metadata_dict = json.load(fid)
            for filename in json_path.parent.glob(json_path.stem + ".*"):
                if filename.suffix.lower() != ".json":
                    metadata_dict["fname"] = str(filename).replace(
                        str(images_dir) + "/", ""
                    )
                    break
            metadata_dicts.append(metadata_dict)

    return metadata_dicts


def write_metadata(metadata_dicts, images_dir):
    """
    Write ULTRRA's image json files.

    :param metadata_dicts: list of metadata json dictionaries
    :param images_dir: path to images directory
    """
    for metadata_dict in metadata_dicts:
        fname = Path(images_dir) / metadata_dict["fname"]
        fname.parent.mkdir(parents=True, exist_ok=True)
        with open(fname.with_suffix(".json"), "w") as fid:
            json.dump(metadata_dict, fid)


def read_origin(sparse_0_dir):
    """
    Read origin.txt. If origin.txt does not exist, return [0.0, 0.0, 0.0]

    :param sparse_0_dir: path to sparse/0 directory
    :return: origin (lat, lon, alt) of ENU coordinate system
    """
    filename = Path(sparse_0_dir) / "origin.txt"
    if filename.exists():
        with open(filename, "r") as fid:
            line = fid.readline()
        origin = [float(x) for x in re.findall(r"[-+]?(?:\d*\.*\d+)", line)]
    else:
        origin = [0.0, 0.0, 0.0]

    return origin


def write_origin(origin, sparse_0_dir):
    """
    Write origin.txt.

    :param origin: origin (lat, lon, alt) of ENU coordinate system
    :param sparse_0_dir: path to sparse/0 directory
    """
    with open(Path(sparse_0_dir) / "origin.txt", "w") as fid:
        fid.write("lat: {}, lon: {}, alt: {}\n".format(*origin))


def compute_centroid(metadata_dicts):
    """
    Compute coordinates (lat, lon, alt) of the centroid for an image collection.

    :param metadata_dicts: list of metadata json dictionaries
    :return: numpy array representing centroid (lat, lon, alt)
    """
    return np.array(
        [
            (
                [
                    metadata_dict["extrinsics"]["lat"],
                    metadata_dict["extrinsics"]["lon"],
                    metadata_dict["extrinsics"]["alt"],
                ]
            )
            for metadata_dict in metadata_dicts
        ]
    ).mean(axis=0)


def map_images(colmap_dir):
    """
    Create csv files that maps indexed rendered images to truth images.
    Outputs train.csv test.csv in COLMAP directory.

    :param colmap_dir: path to directory where COLMAP bin files are located
    """
    # read metadata
    _, images, _ = read_model(colmap_dir, ".bin")

    # map test images
    D = {}
    inds = [image.id for image in images.values() if "test" in image.name]
    if not inds:
        inds = range(1, len(images) + 1, 8)
    D["rendered"] = [f"{i:03d}.png" for i in range(len(inds))]
    D["truth"] = [images[ind].name for ind in inds]
    pd.DataFrame(D).to_csv(Path(colmap_dir) / "test.csv", index=False)

    # map train images
    D = {}
    inds = sorted(list(set(range(1, len(images) + 1)) - set(inds)))
    D["rendered"] = [f"{i:03d}.png" for i in range(len(inds))]
    D["truth"] = [images[ind].name for ind in inds]
    pd.DataFrame(D).to_csv(Path(colmap_dir) / "train.csv", index=False)


def partition_metadata_to_colmap(root_dir, origin=None):
    """
    Converts ULTRRA's image json files to COLMAP's bin files while partitioning into respective cameras.
    Stores bin files in {camera_id}/0/ (will create if it doesn't exist) of root directory.

    :param root_dir: path to root directory
    :param origin: origin (lat, lon, alt) of ENU coordinate system, defaults to centroid of camera positions
    """
    # define directories
    images_dir = Path(root_dir) / "images"

    # read all image json files
    metadata_dicts = read_metadata(images_dir)

    # partition metadata
    cams = sorted(
        list(
            set(
                [
                    json.dumps(metadata_dict["intrinsics"])
                    for metadata_dict in metadata_dicts
                ]
            )
        )
    )
    for i, cam in enumerate(cams):
        print("Converting camera {}...".format(i))
        cameras, images, points3D = metadata_to_model(
            [
                metadata_dict
                for metadata_dict in metadata_dicts
                if json.dumps(metadata_dict["intrinsics"]) == cam
            ],
            origin,
        )
        output_dir = Path(root_dir) / str(i + 1) / "0"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_model(cameras, images, points3D, output_dir, ".bin")


def register_colmap_models(model_moving, model_fixed):
    """
    Registers the moving COLMAP model to the fixed COLMAP model

    :param model_moving: COLMAP dictionary objects (cameras, images, points3D) of moving model
    :param model_fixed: COLMAP dictionary objects (cameras, images, points3D) of fixed model
    :return: COLMAP dictionary objects (cameras, images, points3D) of registered model
    """
    # extract objects
    cameras_moving, images_moving, points3D_moving = model_moving
    cameras_fixed, images_fixed, points3D_fixed = model_fixed

    # find matching images
    names_moving = [image.name for image in images_moving.values()]
    names_fixed = [image.name for image in images_fixed.values()]
    names = sorted(list(set(names_moving) & set(names_fixed)))

    # extract camera positions
    inds_moving = [names_moving.index(name) for name in names]
    inds_fixed = [names_fixed.index(name) for name in names]
    positions_moving = [
        -R.from_quat(np.roll(image.qvec, -1)).inv().apply(image.tvec)
        for image in images_moving.values()
    ]
    positions_fixed = [
        -R.from_quat(np.roll(image.qvec, -1)).inv().apply(image.tvec)
        for image in images_fixed.values()
    ]
    positions_moving = np.stack([positions_moving[ind] for ind in inds_moving])
    positions_fixed = np.stack([positions_fixed[ind] for ind in inds_fixed])

    # perform registration
    d, tform = procrustes(positions_fixed, positions_moving)
    print(
        "The two COLMAP models have a sum of squared differences of {:.4f}.".format(d)
    )
    model_registered = transform_colmap_model(model_moving, tform)

    return model_registered


def procrustes(data1, data2):
    # Adapted from scipy.spatial.procrustes (https://github.com/scipy/scipy/blob/main/scipy/spatial/_procrustes.py)
    # Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions
    # are met:
    #
    # 1. Redistributions of source code must retain the above copyright
    #    notice, this list of conditions and the following disclaimer.
    #
    # 2. Redistributions in binary form must reproduce the above
    #    copyright notice, this list of conditions and the following
    #    disclaimer in the documentation and/or other materials provided
    #    with the distribution.
    #
    # 3. Neither the name of the copyright holder nor the names of its
    #    contributors may be used to endorse or promote products derived
    #    from this software without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    # "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    # A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    # OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    # SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    # LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    Procrustes analysis, a similarity test for two data sets.

    :param data1: Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    :param data2: n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
    :return: float representing disparity; a dict specifying the rotation, scale and translation for the transformation
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    rotation = R.T
    scale = s * norm1 / norm2
    translation = np.mean(data1, 0) - (np.mean(data2, 0).dot(rotation) * scale)

    return disparity, {"rotation": rotation, "scale": scale, "translation": translation}


def transform_colmap_model(model, tform):
    """
    Transforms a COLMAP model

    :param model: COLMAP dictionary objects (cameras, images, points3D) of model
    :param tform: a dict specifying the rotation, scale and translation for the transformation
    :return: COLMAP dictionary objects (cameras, images, points3D) of transformed model
    """
    # extract objects
    cameras, images, points3D = model

    # transform images
    for key, value in images.items():
        r = R.from_quat(np.roll(value.qvec, -1)) * R.from_matrix(tform["rotation"])
        t = tform["scale"] * value.tvec - r.as_matrix() @ tform["translation"]
        images[key] = Image(
            id=value.id,
            qvec=np.roll(r.as_quat(), 1),
            tvec=t,
            camera_id=value.camera_id,
            name=value.name,
            xys=value.xys,
            point3D_ids=value.point3D_ids,
        )

    # transform points3D
    for key, value in points3D.items():
        points3D[key] = Point3D(
            id=value.id,
            xyz=tform["scale"] * value.xyz @ tform["rotation"] + tform["translation"],
            rgb=value.rgb,
            error=value.error,
            image_ids=value.image_ids,
            point2D_idxs=value.point2D_idxs,
        )

    return cameras, images, points3D
