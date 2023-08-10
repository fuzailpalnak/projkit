from typing import List, Tuple
import numpy as np


def _assert_points_3d(points: np.ndarray):
    assert (
        points.shape[-1] == 3
    ), f"Expected points to be of shape Nx3, Received {points.shape}"


def batches(points: List, batch_size: int) -> List[List]:
    """
    Divide a list of points into batches of a specified size.

    Args:
        points (List): List of points to be divided into batches.
        batch_size (int): Size of each batch.

    Returns:
        List[List]: A list of batches, each containing a sublist of points.
    """
    points = np.array(points)
    return np.array_split(points, len(points) // batch_size)


def batch_gen(points: List, batch_size: int) -> Tuple[int, List]:
    """
    Generate batches of points with their corresponding indices.

    Args:
        points (List): List of points to be divided into batches.
        batch_size (int): Size of each batch.

    Yields:
        Tuple[int, List]: A tuple containing the batch index and a sublist of points.
    """

    for i, batch in enumerate(batches(points, batch_size)):
        yield i, batch


def fit_3d_line(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a 3D line to given points and compute the line direction vector.
    https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d

    Args:
        x (np.ndarray): Array of x-coordinates.
        y (np.ndarray): Array of y-coordinates.
        z (np.ndarray): Array of z-coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the line direction vector and the data mean.
    """

    data = np.vstack((x, y, z)).T
    data_mean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - data_mean, full_matrices=False)  # do SVD
    line_direction_vector = vv[0]  # vv[0] is the 1st principal vector
    return line_direction_vector, data_mean


def get_points_on_line_3d(
    points: np.ndarray, lane_direction: np.ndarray, data_mean: np.ndarray
) -> np.ndarray:
    """
    Generate points on a 3D line given its direction vector and data mean.

    Args:
        points (np.ndarray): Array of 3D points.
        lane_direction (np.ndarray): Line direction vector.
        data_mean (np.ndarray): Mean of the input data points.

    Returns:
        np.ndarray: Array of points generated along the 3D line.
    """
    _assert_points_3d(points)
    data = np.vstack((points[:, 0], points[:, 1], points[:, 2])).T
    x_range = data[:, 0].max() - data[:, 0].min()
    y_range = data[:, 1].max() - data[:, 1].min()

    linepts = lane_direction * np.mgrid[-x_range:x_range:15j][:, np.newaxis]
    linepts += data_mean
    return linepts


def fit_and_get_points_on_line_3d(points: np.ndarray) -> np.ndarray:
    """
    Fit a 3D line to given points and generate points on the line.

    Args:
        points (np.ndarray): Array of 3D points.

    Returns:
        np.ndarray: Array of points generated along the fitted 3D line.
    """
    _assert_points_3d(points)
    lane_direction, data_mean = fit_3d_line(points[:, 0], points[:, 1], points[:, 2])
    data = np.vstack((points[:, 0], points[:, 1], points[:, 2])).T
    x_range = data[:, 0].max() - data[:, 0].min()
    y_range = data[:, 1].max() - data[:, 1].min()

    linepts = lane_direction * np.mgrid[-x_range:x_range:15j][:, np.newaxis]
    linepts += data_mean
    return linepts
