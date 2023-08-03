from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class IntrinsicParameters:
    """
    Data class representing intrinsic camera parameters.

    Attributes:
        fx (float): Focal length along the x-axis.
        fy (float): Focal length along the y-axis.
        k1 (float): Distortion coefficient k1.
        k2 (float): Distortion coefficient k2.
        k3 (float): Distortion coefficient k3.
        k4 (float): Distortion coefficient k4.
        Cx (float): Principal point x-coordinate.
        Cy (float): Principal point y-coordinate.
        P1 (float): Tangential distortion coefficient P1.
        P2 (float): Tangential distortion coefficient P2.
        Nx (int): Image width (number of pixels along x-axis).
        Ny (int): Image height (number of pixels along y-axis).
        dx (float): Pixel size along the x-axis in physical units (e.g., mm).
        dy (float): Pixel size along the y-axis in physical units (e.g., mm).
    """

    fx: float
    fy: float
    k1: float
    k2: float
    k3: float
    k4: float
    Cx: float
    Cy: float
    P1: float
    P2: float
    Nx: int
    Ny: int
    dx: float
    dy: float


@dataclass
class ExtrinsicParameters:
    """
    Data class representing extrinsic camera parameters.

    Attributes:
        roll (float): Camera roll angle in degrees.
        pitch (float): Camera pitch angle in degrees.
        yaw (float): Camera yaw angle in degrees.
        camera_center (Tuple[float, float, float]): Camera center coordinates (X, Y, Z).
        direction (Tuple[float, float, float]): Camera viewing direction vector (X, Y, Z).
        up (Tuple[float, float, float]): Camera "up" direction vector (X, Y, Z).
    """

    roll: float
    pitch: float
    yaw: float
    camera_center: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    up: Tuple[float, float, float]


def get_rotation_angles(parameters: ExtrinsicParameters):
    """
    Extract rotation angles (in radians) from the given ExtrinsicParameters object.

    Args:
        parameters (ExtrinsicParameters): Extrinsic camera parameters.

    Returns:
        Tuple[float, float, float]: A tuple containing the rotation angles around the x, y, and z axes (roll, pitch, yaw).
    """

    return (
        np.deg2rad(parameters.roll),
        np.deg2rad(parameters.pitch),
        np.deg2rad(parameters.yaw),
    )


def get_distortion_coefficient(parameters: IntrinsicParameters):
    """
    Get the camera distortion coefficients as an array from given intrinsic parameters.

    Args:
        parameters (IntrinsicParameters): Intrinsic camera parameters.

    Returns:
        np.ndarray: Array containing the camera distortion coefficients.
    """

    return np.array(
        [parameters.k1, parameters.k2, parameters.P1, parameters.P2, parameters.k3]
    )


def get_image_dimension(parameters: IntrinsicParameters):
    """
    Get the camera image dimensions (height, width) from given intrinsic parameters.

    Args:
        parameters (IntrinsicParameters): Intrinsic camera parameters.

    Returns:
        Tuple[int, int]: Tuple containing the image height and width.
    """

    return parameters.Ny, parameters.Nx
