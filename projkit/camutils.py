from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


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


class Query:
    def __init__(self, ic: np.ndarray, wc: np.ndarray):
        self._ic = ic
        self._wc = wc

        self._tree = cKDTree(self._ic)

    def query(self, coordinate, dist_thresh):
        closet_match = None

        dist, pt_idx = self._tree.query(coordinate, k=1)
        if dist <= dist_thresh:
            closet_match = self._wc[pt_idx]

        return dist, closet_match

    def query_batch(self, coordinates, dist_thresh):
        closet_match = None
        dist, pt_idxs = self._tree.query(coordinates, k=1)
        _valid = np.where(np.array(dist) <= dist_thresh)
        closet_match = self._wc[pt_idxs[_valid]]

        return closet_match, dist[_valid]


class CamParam:
    def __init__(self, int_param: IntrinsicParameters, ext_param: ExtrinsicParameters):
        self._int_param = int_param
        self._ext_param = ext_param

    def get_rotation_angles(self):
        return (
            np.deg2rad(self._ext_param.roll),
            np.deg2rad(self._ext_param.pitch),
            np.deg2rad(self._ext_param.yaw),
        )

    def get_5_distortion_coefficient(self):
        return np.array(
            [
                self._int_param.k1,
                self._int_param.k2,
                self._int_param.P1,
                self._int_param.P2,
                self._int_param.k3,
            ]
        )

    def get_image_dimension(self):
        return self._int_param.Ny, self._int_param.Nx


def _filter(w, h, ic) -> np.ndarray:
    """
    Internal function to filter image coordinates based on their position within the image frame.

    Args:
        w (int): Width of the image frame.
        h (int): Height of the image frame.
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.

    Returns:
        np.ndarray: Array of indices corresponding to the valid points within the image frame.
    """

    indices = np.where(
        (ic.T[0, :] <= w) & (ic.T[0, :] >= 0) & (ic.T[1, :] <= h) & (ic.T[1, :] >= 0)
    )[0]
    return indices


def filter_image_points_with_img_dim(w, h, ic) -> np.ndarray:
    """
    Filter image coordinates to keep only the points lying within the specified image dimensions.

    Args:
        w (int): Width of the image frame.
        h (int): Height of the image frame.
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.

    Returns:
        np.ndarray: Filtered image coordinates, containing only the points within the image frame.
    """

    indices = _filter(w, h, ic)
    return ic[indices]


def filter_image_and_world_points_with_img_dim(
    w: int, h: int, ic: np.ndarray, wc: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter image and world coordinates to keep only the points lying within the specified image dimensions.

    Args:
        w (int): Width of the image frame.
        h (int): Height of the image frame.
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.
        wc (np.ndarray): 2D array of shape (Nx3) containing world coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing filtered image coordinates and corresponding world coordinates.
    """

    assert ic.shape == wc.shape, (
        f"Expected image_coordinates and world_coordinates to have same shape, Got "
        f"image_coordinates = {ic.shape}, world_coordinates = {wc.shape}"
    )
    indices = _filter(w, h, ic)

    return ic[indices], wc[indices]


def generate_image(w: int, h: int, ic: np.ndarray, wc: np.ndarray = None) -> np.ndarray:
    """
    Generate an image frame with points marked at their respective positions.

    Args:
        w (int): Width of the image frame.
        h (int): Height of the image frame.
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.
        wc (np.ndarray, optional): 2D array of shape (Nx3) containing world coordinates.
            If provided, the value of each point in the image will be set based on its corresponding world coordinate's z-value.

    Returns:
        np.ndarray: A 2D array representing the image frame with points marked at their respective positions.
    """

    assert ic.shape == wc.shape, (
        f"Expected image_coordinates and world_coordinates to have same shape, Got "
        f"image_coordinates = {ic.shape}, world_coordinates = {wc.shape}"
    )
    frame = np.zeros((h, w))
    frame[ic[:, 1], ic[:, 0]] = 255 if wc is None else wc[:, -1]
    return frame


def nn_search(ic: np.ndarray, wc: np.ndarray):
    kd_tree = cKDTree(ic)
