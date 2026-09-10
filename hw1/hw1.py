"""HW1: transform the four vehicle corners into the world frame."""

import numpy as np


def get_corners(xy, theta, corner1, corner2, corner3, corner4):
    """Return the four world-frame corner positions in the same order.

    Args:
        xy: Vehicle position as a NumPy array of shape (2, 1).
        theta: Vehicle heading in radians, measured counterclockwise.
        corner1, corner2, corner3, corner4: Vehicle-frame corner positions,
            each with shape (2, 1).

    Returns:
        A tuple of four NumPy arrays, each with shape (2, 1).
    """
    # TODO: implement the rotation and translation for each corner.
    # raise NotImplementedError("Implement get_corners before running the checks.")
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                 [np.sin(theta), np.cos(theta)]
                 ])
    
    c1 = np.matmul(R, corner1) + xy
    c2 = np.matmul(R, corner2) + xy
    c3 = np.matmul(R, corner3) + xy
    c4 = np.matmul(R, corner4) + xy

    return (c1, c2, c3, c4)
