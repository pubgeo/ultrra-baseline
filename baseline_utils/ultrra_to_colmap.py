"""
TODO ULTRRA CONTEST HEADER
"""

import argparse
from pathlib import Path

from .read_write_model import write_model
from .utils import (
    compute_centroid,
    metadata_to_model,
    read_metadata,
    write_origin,
)


def ultrra_to_colmap(root_dir, origin=None, camera_model="FULL_OPENCV", save_ext=".bin"):
    """
    Converts ULTRRA's image json files to COLMAP's bin files.
    Expects json files in images/ of root directory.
    Stores bin files in sparse/0/ (will create if it doesn't exist) of root directory.
    Also stores origin.txt in sparse/0/ to use for conversion back.

    :param root_dir: path to root directory
    :param origin: origin (lat, lon, alt) of ENU coordinate system, defaults to centroid of camera positions
    """
    # define directories
    images_dir = Path(root_dir) / "images"
    sparse_0_dir = Path(root_dir) / "sparse" / "0"

    # read ULTRRA files
    metadata_dicts = read_metadata(images_dir)

    # convert metadata
    if not origin:
        origin = compute_centroid(metadata_dicts)
    cameras, images, points3D = metadata_to_model(metadata_dicts, origin, camera_model=camera_model)

    # write files
    sparse_0_dir.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images, points3D, sparse_0_dir, save_ext)
    write_origin(origin, sparse_0_dir)


def main():
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, required=True, help="root directory of dataset"
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        help="origin (lat, lon, alt) of ENU coordinate system",
        default=None,
    )
    args = parser.parse_args()

    # convert metadata
    ultrra_to_colmap(args.root_dir, args.origin)


if __name__ == "__main__":
    main()
