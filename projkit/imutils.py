from typing import Callable, Any, List, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree


def _assert_ic_wc_shape(ic: np.ndarray, wc: np.ndarray):
    assert ic.shape[0] == wc.shape[0], (
        f"Expected image_coordinates and world_coordinates to have same number of points, Got "
        f"ic = {ic.shape[0]}, wc = {wc.shape[0]}"
    )

    assert ic.shape[-1] == 2, (
        f"Expected image_coordinates be of dim (Nx2), Got " f"ic = {ic.shape}"
    )
    assert wc.shape[-1] == 3, (
        f"Expected world_coordinates be of dim (Nx3), Got " f"wc = {wc.shape}"
    )


class Query:
    """
    Class for performing nearest-neighbor queries on image and world coordinates.

    Args:
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.
        wc (np.ndarray): 2D array of shape (Nx3) containing world coordinates.

    Attributes:
        _ic (np.ndarray): Image coordinates.
        _wc (np.ndarray): World coordinates.
        _tree (cKDTree): KD-tree for efficient nearest-neighbor search on image coordinates.

    Methods:
        query(coordinates: list, dist_thresh: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Perform a nearest-neighbor query for given coordinates and distance threshold.
        generate_points_for_nn_search(Ny: int, Nx: int, binary_mask: np.ndarray) -> List[Tuple[int, int]]:
            Generate a list of image coordinates for nearest-neighbor search based on a binary mask.
    """

    def __init__(self, ic: np.ndarray, wc: np.ndarray):
        """
        Initialize the Query object.

        Args:
            ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.
            wc (np.ndarray): 2D array of shape (Nx3) containing world coordinates.
        """

        _assert_ic_wc_shape(ic, wc)

        self._ic = ic
        self._wc = wc

        self._tree = cKDTree(self._ic.astype(np.int))

    def query(
        self, coordinates: list, dist_thresh: int, k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a nearest-neighbor query (w, h) for given coordinates and distance threshold.

        Args:
            coordinates (list): List of query coordinates.
            dist_thresh (int): Maximum distance for considering nearest neighbors.
            k (int): The list of k-th nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing arrays of image coordinates,
            world coordinates, and distances for the nearest neighbors within the specified threshold.
        """

        c_ic, c_wc, c_dist = None, None, None

        dist, pt_idxs = self._tree.query(coordinates, k=k)
        _valid = np.where(np.array(dist) <= dist_thresh)
        if len(_valid[0]) > 0:
            c_ic, c_wc, c_dist = self._ic[_valid], self._wc[_valid], dist[_valid]

        return c_ic, c_wc, c_dist

    def generate_points_for_nn_search(
        self, Ny: int, Nx: int, binary_mask: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Generate a list of image coordinates for nearest-neighbor search based on a binary mask.

        Args:
            Ny (int): Height of the image frame.
            Nx (int): Width of the image frame.
            binary_mask (np.ndarray): 2D binary mask representing regions of interest.

        Returns:
            List[Tuple[int, int]]: A list of image coordinates for nearest-neighbor search.
        """

        _missing_z_values_image = difference(Ny, Nx, self._ic, self._wc, binary_mask)
        x, y = np.where(_missing_z_values_image == 255)
        return list(zip(y, x))


def to_array(h: int, w: int, ic: np.ndarray, wc: np.ndarray) -> np.ndarray:
    """
    Convert image and world coordinates to a 3D array (frame) of size (h, w, 3).

    Args:
        h (int): Height of the resulting 3D array (frame).
        w (int): Width of the resulting 3D array (frame).
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.
        wc (np.ndarray): 2D array of shape (Nx3) containing world coordinates.

    Returns:
        np.ndarray: A 3D array (frame) of size (h, w, 3) with world coordinates placed at their respective image
        positions.
    """
    _assert_ic_wc_shape(ic, wc)

    ic = ic.astype(np.int)

    frame = np.zeros((h, w, 3))
    frame[ic[:, 1], ic[:, 0]] = wc

    return frame


def to_image(h: int, w: int, ic: np.ndarray, wc: np.ndarray) -> np.ndarray:
    """
    Convert 2D image coordinates to an image frame of size (h, w) with optional z-values.

    Args:
        h (int): Height of the image frame.
        w (int): Width of the image frame.
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.
        wc (np.ndarray, optional): 2D array of shape (Nx3) containing world coordinates. If provided, the value of each
        point in the image will be set based on its corresponding world coordinate's z-value. If not provided, the image
        frame will be set to 255 at the specified image coordinates.

    Returns:
        np.ndarray: A 2D array representing the image frame with points marked at their respective positions.
    """
    _assert_ic_wc_shape(ic, wc)

    ic = ic.astype(np.int)

    frame = np.zeros((h, w))
    frame[ic[:, 1], ic[:, 0]] = 255 if wc is None else wc[:, -1]
    return frame


def intersection(
    binary_mask: np.ndarray, ic: np.ndarray, wc: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Compute the intersection points between a binary mask and given image and world coordinates.

    Args:
        binary_mask (np.ndarray): 2D binary mask representing regions of interest.
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.
        wc (np.ndarray): 2D array of shape (Nx3) containing world coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two items:
            - intersection_frame (np.ndarray): A 3D array of shape (h, w, 3) representing an intersection frame.
            - locations (np.ndarray): A 2D array containing the intersection points with non-zero z-coordinates.
    """

    _assert_ic_wc_shape(ic, wc)

    h, w = binary_mask.shape
    intersection_frame = to_array(h, w, ic, wc)
    intersection_frame[np.where(binary_mask != 255)] = (0, 0, 0)
    locations = intersection_frame[np.where(intersection_frame[:, :, -1] > 0)]

    return intersection_frame, locations


def cluster_binary_using_contour(binary_mask: np.ndarray) -> List[np.ndarray]:
    """
    Cluster regions in the binary mask using contour detection.

    Args:
        binary_mask (np.ndarray): 2D binary mask representing the regions of interest.

    Returns:
        List[np.ndarray]: A list containing clustered regions' image coordinates as NumPy arrays.
    """

    cluster_image_coordinates = list()
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        _tmp = np.zeros_like(binary_mask)
        # Calculate the area of the current contour
        contour_area = cv2.contourArea(contour)

        # Set the contour to 0 (remove it) if its area is below the threshold
        cv2.drawContours(_tmp, [contour], 0, 255, -1)

        x, y = np.where(_tmp == 255)
        cluster_image_coordinates.append(np.column_stack((x, y)))
    return cluster_image_coordinates


def intersection_on_clusters(
    fn: Callable[[Any], Any], binary_mask: np.ndarray, ic: np.ndarray, wc: np.ndarray
) -> Tuple[list, list]:
    """
    Compute the intersection points on clusters found in the binary mask, given image and world coordinates.

    Args:
        fn (function): A function that takes a binary mask as input and returns a list of cluster locations.
        binary_mask (np.ndarray): 2D binary mask representing clustered regions.
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.
        wc (np.ndarray): 2D array of shape (Nx3) containing world coordinates.

    Returns:
        Tuple[list, list]: A tuple containing two lists. The first list contains the cluster locations in image
        coordinates,
        and the second list contains the cluster locations in world coordinates, excluding points with z-coordinate
         equal to 0.
    """

    cluster_locations_wc = list()

    cluster_locations_ic = fn(binary_mask)
    image3d, locations = intersection(binary_mask, ic, wc)
    for i, cl in enumerate(cluster_locations_ic):
        _tmp = image3d[cl[:, 0], cl[:, 1]]
        _tmp = _tmp[np.where(_tmp[:, -1] != 0)]
        if _tmp.size != 0:
            cluster_locations_wc.append(_tmp)
    return cluster_locations_ic, cluster_locations_wc


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
        Tuple[np.ndarray, np.ndarray]: A tuple containing filtered image coordinates and corresponding world
        coordinates.
    """

    _assert_ic_wc_shape(ic, wc)

    indices = _filter(w, h, ic)

    wc = wc[indices]

    return ic[indices], wc, wc[:, -1:]


def difference(
    Ny: int, Nx: int, ic: np.ndarray, wc: np.ndarray, binary_mask: np.ndarray
) -> np.ndarray:
    """
    Compute the difference between binary mask and projected point cloud highlighting missing z-values in the
    point cloud for areas where mask is available.

    Args:
        Ny (int): Height of the image frame.
        Nx (int): Width of the image frame.
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.
        wc (np.ndarray): 2D array of shape (Nx3) containing world coordinates.
        binary_mask (np.ndarray): 2D binary mask representing regions of interest.

    Returns:
        np.ndarray: A 2D array representing the difference image that highlights missing z-values in the point cloud.
    """
    assert (
        binary_mask.ndim == 2
    ), f"Expected binary mask to have shape (HxW), Recievec {binary_mask.shape}"

    _assert_ic_wc_shape(ic, wc)

    point_cloud_image = to_image(Ny, Nx, ic, wc)

    _z_value_image = np.where(binary_mask == 255, point_cloud_image, 0)
    _missing_z_values_image = np.where(binary_mask, _z_value_image, 255)
    _missing_z_values_image = np.where(_missing_z_values_image > 0, 0, 255)

    # _combined_image = _z_value_image + _missing_z_values_image

    return _missing_z_values_image


def nn_interpolation(ic: np.ndarray, wc: np.ndarray) -> Query:
    """
    Perform nearest-neighbor interpolation using image and world coordinates.

    Args:
        ic (np.ndarray): 2D array of shape (Nx2) containing image coordinates.
        wc (np.ndarray): 2D array of shape (Nx3) containing world coordinates.

    Returns:
        Query: A Query object initialized with image coordinates (ic) and world coordinates (wc).
    """

    _assert_ic_wc_shape(ic, wc)

    return Query(ic, wc)
