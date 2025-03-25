import numpy as np


def rotate_thickness_matrix(old_mat: np.ndarray, rotation: int) -> np.ndarray:
    """
    Rotate a 2x2 thickness matrix by `rotation * 90` degrees clockwise,
    preserving the mapping:
      (0,0)=bottom, (0,1)=top, (1,0)=left, (1,1)=right
    so that after rotation they still correspond to 
      (0,0)=new bottom, (0,1)=new top, (1,0)=new left, (1,1)=new right.
    """
    # For convenience, define short aliases
    B = old_mat[0, 0]  # bottom
    T = old_mat[0, 1]  # top
    L = old_mat[1, 0]  # left
    R = old_mat[1, 1]  # right

    new_mat = np.zeros((2, 2), dtype=old_mat.dtype)

    # Normalize rotation so it's 0,1,2,3
    rotation = rotation % 4

    if rotation == 0:
        # 0° means "no rotation"
        new_mat[0, 0] = B
        new_mat[0, 1] = T
        new_mat[1, 0] = L
        new_mat[1, 1] = R

    elif rotation == 1:
        # 90° CCW
        # old bottom -> new right
        # old right -> new top
        # old top -> new left
        # old left -> new bottom
        new_mat[0, 0] = L
        new_mat[0, 1] = R
        new_mat[1, 0] = T
        new_mat[1, 1] = B

    elif rotation == 2:
        # 180° CCW
        # old bottom -> new top
        # old top -> new bottom
        # old left -> new right
        # old right -> new left
        new_mat[0, 0] = T
        new_mat[0, 1] = B
        new_mat[1, 0] = R
        new_mat[1, 1] = L

    else:  # rotation == 3
        # 270° CCW
        new_mat[0, 0] = R  # old right -> new bottom
        new_mat[0, 1] = L  # old left -> new top
        new_mat[1, 0] = B  # old bottom -> new left
        new_mat[1, 1] = T  # old top -> new right

    return new_mat


def rotation_sequence(rotation):
    if rotation == 0:
        return [[0, 0], [0, 1], [1, 0], [1, 1]]
    elif rotation == 1:
        return [[1, 1], [1, 0], [0, 0], [0, 1]]
    elif rotation == 2:
        return [[0, 1], [0, 0], [1, 1], [1, 0]]
    elif rotation == 3:
        return [[1, 0], [1, 1], [0, 1], [0, 0]]
