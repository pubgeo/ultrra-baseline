import numpy as np
from scipy.linalg import orthogonal_procrustes, norm
from numpy.random import default_rng
rng = default_rng()


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


def ransac_procrustes(data1, data2, min_num_points=4, max_iters=1000, threshold=1, num_close_points=2):
    """
    Procrustes analysis with RANdom SAmple Consensus (RANSAC)

    :param data1: Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    :param data2: n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
    :param min_num_points: minimum number of points required to perform Procrustes analysis
    :param max_iters: maximum number of iterations to run RANSAC
    :param thershold: threshold value to differentiate between inliers and outliers
    :param num_close_points: number of additional inliers required to assert model
    :return: float representing disparity; a dict specifying the rotation, scale and translation for the transformation
    """
    d_best, tform_best = np.inf, {}
    for _ in range(max_iters):
        ids = rng.permutation(data1.shape[0])

        maybe_inliers = ids[: min_num_points]
        d, tform = procrustes(data1[maybe_inliers], data2[maybe_inliers])
        data2_tformed = tform["scale"] * data2[ids][min_num_points :] @ tform["rotation"] + tform["translation"]

        thresholded = norm(data2_tformed-data1[ids][min_num_points :], axis=1) < threshold

        inlier_ids = ids[min_num_points :][np.flatnonzero(thresholded).flatten()]

        if inlier_ids.size > num_close_points:
            inlier_points = np.hstack([maybe_inliers, inlier_ids])
            d, tform = procrustes(data1[inlier_points], data2[inlier_points])

            if d < d_best:
                d_best = d
                tform_best = tform
    
    return d_best, tform_best


if __name__ == "__main__":
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    Y = np.array([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, -1, 1], [-1, 0, 1], [-1, -1, 0], [-2, -2, 2]])

    d, tform = procrustes(X, Y)
    print(d)
    print(tform)

    d, tform = ransac_procrustes(X, Y)
    print(d)
    print(tform)
