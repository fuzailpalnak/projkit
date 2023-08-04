from typing import Any, Tuple

import cv2
import numpy as np


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """
    Convert points in 2D to homogeneous coordinates.

    Args:
        points (np.ndarray): 2D array containing points in Cartesian coordinates.

    Returns:
        np.ndarray: 2D array with points in homogeneous coordinates.
    """

    assert points.ndim == 2, f"Expected points to have dim = 2, received {points.ndim}"

    n, d = points.shape
    return np.c_[points, np.ones(n)]


def de_homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points from homogeneous coordinates to Cartesian coordinates.

    Args:
        points (np.ndarray): 2D array containing points in homogeneous coordinates.

    Returns:
        np.ndarray: 2D array with points in Cartesian coordinates.
    """

    assert points.ndim == 2, f"Expected points to have dim = 2, received {points.ndim}"
    _dh = points[:, 0:-1] / points[:, -1][:, np.newaxis]
    return _dh


def get_t_from_R_C(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Calculate the translation vector t from given rotation matrix R and camera center C.

    Args:
        R (np.ndarray): 3x3 rotation matrix representing camera orientation.
        C (np.ndarray): Camera center coordinates as a 1D array (shape: (3,)).

    Returns:
        np.ndarray: 3x1 translation vector representing camera position.
    """
    return -R @ C.reshape(3, 1)


def get_P_from_K_R_t(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calculate the projection matrix P = K[R|t] from intrinsic matrix K, rotation matrix R, and translation vector t.

    Args:
        K (np.ndarray): 3x3 intrinsic camera matrix.
        R (np.ndarray): 3x3 rotation matrix representing camera orientation.
        t (np.ndarray): 3x1 translation vector representing camera position.

    Returns:
        np.ndarray: 3x4 projection matrix representing the full camera projection.
    """
    return K @ np.hstack([R, t])


def get_P(int_mat: np.ndarray, ext_mat: np.ndarray):
    """
    Calculate the projection matrix P = K[R|t] from intrinsic matrix and extrinsic matrix.

    Args:
        int_mat (np.ndarray): 3x3 intrinsic camera matrix.
        ext_mat (np.ndarray): 3x4 extrinsic matrix representing camera pose.

    Returns:
        np.ndarray: 3x4 projection matrix representing the full camera projection.
    """
    return int_mat @ ext_mat


def get_C_from_R_t(R, t):
    """
    Calculate the origin of the coordinate system (camera center) C from given rotation matrix R and translation vector t.

    Args:
        R (np.ndarray): 3x3 rotation matrix representing camera orientation.
        t (np.ndarray): 3x1 translation vector representing camera position.

    Returns:
        np.ndarray: 3x1 camera center coordinates.
    """
    return -(R.T @ t)


def project_in_3d_with_K_R_t_scale(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, s: np.ndarray, ic: np.ndarray
) -> np.ndarray:
    """
    Project 2D image coordinates to 3D world coordinates with scale factor s, intrinsic matrix K, rotation matrix R,
    and translation vector t.

    Args:
        K (np.ndarray): 3x3 intrinsic camera matrix.
        R (np.ndarray): 3x3 rotation matrix representing camera orientation.
        t (np.ndarray): 3x1 translation vector representing camera position.
        s (np.ndarray): Scale factor applied to the image coordinates.
        ic (np.ndarray): 2D array of shape (Nx2) containing 2D image coordinates.

    Returns:
        np.ndarray: 2D array of shape (Nx3) containing 3D points in world coordinates.
    """

    return np.linalg.inv(R) @ ((np.linalg.inv(K) @ (s * to_homogeneous(ic)).T) - t)


def project_in_2d_with_K_R_t(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, wc: np.ndarray
) -> np.ndarray:
    """
    Project 3D points in world coordinates to 2D image coordinates using intrinsic matrix K, rotation matrix R, and
    translation vector t.

    Args:
        K (np.ndarray): 3x3 intrinsic camera matrix.
        R (np.ndarray): 3x3 rotation matrix representing camera orientation.
        t (np.ndarray): 3x1 translation vector representing camera position.
        wc (np.ndarray): 2D array of shape (Nx3) containing 3D points in world coordinates.

    Returns:
        np.ndarray: 2D array of shape (Nx2) containing 2D points in image coordinates.
    """

    assert (
        wc.shape[-1] == 3 and wc.ndim == 2
    ), f"Expected shape for wc = (Nx3), received {wc.shape}"

    wc_homogeneous = to_homogeneous(wc)
    ic = de_homogenize((get_P_from_K_R_t(K, R, t) @ wc_homogeneous.T).T)

    return ic


def get_ic_wc_z_from_proj(inp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract image coordinates (ic), world coordinates (wc), and z-coordinates from the given projection array.

    Args:
        inp (np.ndarray): 2D array with shape (Nx3) representing a projection array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the image coordinates (ic),
        world coordinates (wc), and z-coordinates extracted from the input projection array.
    """
    assert (
        inp.ndim == 2 and inp.shape[-1] == 5
    ), f"Expected inp to have shape (Nx5), Got {inp.shape}"
    return inp[:, :2], inp[:, 2:], inp[:, -1:]


def project_in_2d_with_K_R_t_dist_coeff(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    d: np.ndarray,
    wc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points in world coordinates to 2D image coordinates with distortion correction.

    Args:
        K (np.ndarray): 3x3 intrinsic camera matrix.
        R (np.ndarray): 3x3 rotation matrix representing camera orientation.
        t (np.ndarray): 3x1 translation vector representing camera position.
        d (np.ndarray): Distortion coefficients.
        wc (np.ndarray): 2D array of shape (Nx3) containing 3D points in world coordinates.

    Returns:
        np.ndarray: 2D array of shape (Nx5) containing 2D points in image coordinates and its corresponding 3d coordinates
         with distortion correction.
    """

    # https://answers.opencv.org/question/20138/projectpoints-fails-with-points-behind-the-camera/

    assert (
        wc.shape[-1] == 3 and wc.ndim == 2
    ), f"Expected shape for wc = (Nx3), received {wc.shape}"

    wc_homogeneous = to_homogeneous(wc)
    cc = (np.hstack([R, t]) @ wc_homogeneous.T).T
    z = cc[:, -1]

    cc = de_homogenize(cc)
    cc = cv2.undistortPoints(cc[np.newaxis, :, :], K, d, P=K).squeeze()

    # Remove points which lie behind the camera
    cc = cc[z > 0]

    # Project from camera coordinates to image coordinates
    ic = (K @ to_homogeneous(cc).T).T
    ic = de_homogenize(ic)

    return get_ic_wc_z_from_proj(np.hstack([ic, wc[z > 0]]))


def get_int_mat(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Generate the camera intrinsic matrix (3x3) from focal lengths (fx, fy) and principal point (cx, cy).

    Args:
        fx (float): Focal length along the x-axis.
        fy (float): Focal length along the y-axis.
        cx (float): Principal point x-coordinate.
        cy (float): Principal point y-coordinate.

    Returns:
        np.ndarray: 3x3 intrinsic matrix representing the camera parameters.
    """

    return np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )


def get_ext_mat(R: np.ndarray, t: np.ndarray):
    """
    Generate the camera extrinsic matrix (3x4) from rotation matrix R and translation vector t.

    Args:
        R (np.ndarray): 3x3 rotation matrix representing camera orientation.
        t (np.ndarray): 3x1 translation vector representing camera position.

    Returns:
        np.ndarray: 3x4 extrinsic matrix representing the camera pose.
    """

    return np.hstack([R, t])


def Rx(rx: float):
    """
    Generate a 3x3 rotation matrix around the x-axis.

    Args:
        rx (float): Rotation angle in radians around the x-axis.

    Returns:
        np.ndarray: 3x3 rotation matrix representing the rotation around the x-axis.
    """

    return np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )


def Ry(ry: float):
    """
    Generate a 3x3 rotation matrix around the y-axis.

    Args:
        ry (float): Rotation angle in radians around the y-axis.

    Returns:
        np.ndarray: 3x3 rotation matrix representing the rotation around the y-axis.
    """

    return np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )


def Rz(rz: float):
    """
    Generate a 3x3 rotation matrix around the z-axis.

    Args:
        rz (float): Rotation angle in radians around the z-axis.

    Returns:
        np.ndarray: 3x3 rotation matrix representing the rotation around the z-axis.
    """

    return np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )


def get_R(rx: float, ry: float, rz: float, to_rad=True) -> np.ndarray:
    """
    Calculate the 3x3 camera rotation matrix from the given rotation angles around x, y, and z axes.

    Args:
        rx (float): Rotation angle in radians around the x-axis.
        ry (float): Rotation angle in radians around the y-axis.
        rz (float): Rotation angle in radians around the z-axis.

    Returns:
        np.ndarray: 3x3 rotation matrix representing the camera orientation.
    """

    # https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20162017/LectureNotes/CV/PinholeCamera/PinholeCamera.html
    if to_rad:
        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
        rz = np.deg2rad(rz)

    return (Rz(rz) @ Ry(ry) @ Rx(rx)).T
