from typing import List, Tuple


def batches(points: List, batch_size: int) -> List[List]:
    """
    Divide a list of points into batches of a specified size.

    Args:
        points (List): List of points to be divided into batches.
        batch_size (int): Size of each batch.

    Returns:
        List[List]: A list of batches, each containing a sublist of points.
    """

    return [points[i : i + batch_size] for i in range(0, len(points), batch_size)]


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
