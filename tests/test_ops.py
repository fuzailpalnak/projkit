import numpy as np

from projkit.ops import de_homogenize, to_homogeneous


def test_de_homogenize():
    # Test case : Simple case with 3 points
    input_points = np.array([[2, 4, 2, 2], [6, 8, 2, 2], [9, 3, 3, 3]])
    expected_output = np.array([[1, 2, 1], [3, 4, 1], [3, 1, 1]])
    assert np.allclose(de_homogenize(input_points), expected_output)

    # Test case : Case with negative coordinates
    input_points = np.array([[-2, -4, 2, 2], [-6, -8, 2, 2], [-9, 3, 3, 3]])
    expected_output = np.array([[-1, -2, 1], [-3, -4, 1], [-3, 1, 1]])
    assert np.allclose(de_homogenize(input_points), expected_output)

    # Test case : Case with single point
    input_points = np.array([[0, 0, 1, 1]])
    expected_output = np.array([[0, 0, 1]])
    assert np.allclose(de_homogenize(input_points), expected_output)

    # Test case : Simple case with 3 points
    input_points = np.array([[2, 4, 2], [6, 8, 2], [9, 3, 3]])
    expected_output = np.array([[1, 2], [3, 4], [3, 1]])
    assert np.allclose(de_homogenize(input_points), expected_output)

    # Test case : Case with points at infinity
    input_points = np.array([[3, 6, 0], [9, 12, 0], [0, 0, 0]])
    expected_output = np.array([[np.inf, np.inf], [np.inf, np.inf], [np.nan, np.nan]])
    assert np.allclose(de_homogenize(input_points), expected_output, equal_nan=True)

    # Test case : Case with negative coordinates
    input_points = np.array([[-2, -4, 2], [-6, -8, 2], [-9, 3, 3]])
    expected_output = np.array([[-1, -2], [-3, -4], [-3, 1]])
    assert np.allclose(de_homogenize(input_points), expected_output)

    # Test case : Case with single point
    input_points = np.array([[0, 0, 1]])
    expected_output = np.array([[0, 0]])
    assert np.allclose(de_homogenize(input_points), expected_output)


def test_to_homogeneous():
    # Test case 1: Basic case with 2D points
    points1 = np.array([[1, 2], [3, 4], [5, 6]])
    expected_output1 = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1]])
    assert np.array_equal(to_homogeneous(points1), expected_output1)

    # Test case 2: 2D points with negative coordinates
    points2 = np.array([[-2, 3], [4, -5], [-6, -7]])
    expected_output2 = np.array([[-2, 3, 1], [4, -5, 1], [-6, -7, 1]])
    assert np.array_equal(to_homogeneous(points2), expected_output2)

    # Test case 3: 2D points with zero coordinates
    points3 = np.array([[0, 0], [0, 0], [0, 0]])
    expected_output3 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    assert np.array_equal(to_homogeneous(points3), expected_output3)

    # Test case 4: 2D points with non-integer coordinates
    points4 = np.array([[1.5, 2.3], [3.1, 4.7], [5.9, 6.2]])
    expected_output4 = np.array([[1.5, 2.3, 1], [3.1, 4.7, 1], [5.9, 6.2, 1]])
    assert np.array_equal(to_homogeneous(points4), expected_output4)

    # Test case 5: Single point
    points5 = np.array([[10, 20]])
    expected_output5 = np.array([[10, 20, 1]])
    assert np.array_equal(to_homogeneous(points5), expected_output5)

    # Test case 6: Empty input
    points6 = np.array([])
    try:
        to_homogeneous(points6)
    except AssertionError:
        assert True
    else:
        assert False, "Empty input should raise an AssertionError"

    # Test case 7: Input with more than 2 dimensions
    points7 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    try:
        to_homogeneous(points7)
    except AssertionError:
        assert True
    else:
        assert False, "Input with more than 2 dimensions should raise an AssertionError"

    # Test case 8: Nx3 points
    points8 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_output8 = np.array([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])
    assert np.array_equal(to_homogeneous(points8), expected_output8)

    # Test case 9: Nx3 points with non-integer coordinates
    points9 = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
    expected_output9 = np.array(
        [[1.1, 2.2, 3.3, 1], [4.4, 5.5, 6.6, 1], [7.7, 8.8, 9.9, 1]]
    )
    assert np.array_equal(to_homogeneous(points9), expected_output9)
