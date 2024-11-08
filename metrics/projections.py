import typing
import numpy as np
import pyproj

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    _USE_SCIPY = False
else:
    _USE_SCIPY = True


def _rotation_matrix(axis: str, angle_radians: float) -> np.ndarray:
    """Create a simple rotation matrix around a specified axis.

    Args:
        axis (str): the axis to rotate around (must be x, y, or z, case-sensitive)
        angle (float): the angle to rotate through, specified in radians

    Returns:
        np.ndarray: a 3x3 numpy array, suitable for matrix multiplication
    """
    sth = np.sin(angle_radians)
    cth = np.cos(angle_radians)
    if axis == "x":
        return np.array(
            [
                [1, 0, 0],
                [0, cth, -sth],
                [0, sth, cth],
            ]
        )
    if axis == "y":
        return np.array(
            [
                [cth, 0, sth],
                [0, 1, 0],
                [-sth, 0, cth],
            ]
        )
    if axis == "z":
        return np.array(
            [
                [cth, -sth, 0],
                [sth, cth, 0],
                [0, 0, 1],
            ]
        )


def from_euler_esque(
    axes: str, angles: typing.Collection[float], degrees: bool = True
) -> np.ndarray:
    """Non-scipy substitution for Rotation.from_euler.as_matrix

    This does not take advantage of scipy's math boosts, and is likely to perform more
    slowly than its equivalent. However, it does not have a scipy dependency, and this
    might make it preferable for certain applications.

    Args:
        axes (str): order of axial rotations as Euler - case-insensitive
        angles (typing.Collection[float]): the angles to rotate each axis through
        degrees (bool, optional): indicates if the given angles are in degrees (not radians). Defaults to True.

    Returns:
        np.ndarray: a 3x3 numpy array, suitable for matrix multiplication
    """
    arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # to do the same thing as the from_euler function, we need to reverse the order of
    # arguments in matrix multiplication
    for axis, _angle in zip(axes.lower()[::-1], angles[::-1]):
        if degrees:
            angle = np.radians(_angle)
        else:
            angle = _angle
        arr = arr.dot(_rotation_matrix(axis, angle))
    return arr



def geodetic_to_enu(
    lat, lon, alt, lat_org, lon_org, alt_org, use_scipy: bool = _USE_SCIPY
):
    """
    convert LLA to ENU
    :params lat, lon, alt: input LLA coordinates
    :params lat_org, lon_org, alt_org: LLA of the origin of the local ENU coordinate system
    :return: east, north, up coordinate
    """
    transformer = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=True,
    )
    x, y, z = transformer.transform(lon, lat, alt, radians=False)
    x_org, y_org, z_org = transformer.transform(
        lon_org, lat_org, alt_org, radians=False
    )
    vec = np.array([[x - x_org, y - y_org, z - z_org]]).T
    if use_scipy:
        rot1 = Rotation.from_euler(
            "x", -(90 - lat_org), degrees=True
        ).as_matrix()  # angle*-1 : left handed *-1
        rot3 = Rotation.from_euler(
            "z", -(90 + lon_org), degrees=True
        ).as_matrix()  # angle*-1 : left handed *-1
    else:
        rot1 = from_euler_esque("x", -(90 - lat_org), degrees=True)
        rot3 = from_euler_esque("z", -(90 + lon_org), degrees=True)
    rotMatrix = rot1.dot(rot3)
    enu = rotMatrix.dot(vec).T.ravel()
    return enu.T
