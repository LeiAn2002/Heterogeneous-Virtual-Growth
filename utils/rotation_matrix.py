import numpy as np
from itertools import permutations, product


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


#  idx 0→Z-  1→Z+  2→Y-  3→Y+  4→X-  5→X+

FACES_3D = np.array([
    [0,  0, -1],   # Z-  (bottom)
    [0,  0,  1],   # Z+
    [0, -1,  0],   # Y-  (back)
    [0,  1,  0],   # Y+
    [-1,  0,  0],   # X-  (left)
    [1,  0,  0],   # X+
], dtype=int)


def _generate_rot_mats():
    """Enumerate all 24 proper rotations of the cube."""
    mats = []
    for perm in permutations([0, 1, 2]):            # axis permutation
        P = np.eye(3, dtype=int)[:, perm]           # permutation matrix
        for signs in product([-1, 1], repeat=3):    # sign flips
            S = np.diag(signs)
            R = P @ S                               # candidate
            if np.linalg.det(R) == 1:               # proper (right-handed)
                mats.append(R.astype(int))
    # sanity check
    mats_unique = np.unique(np.stack(mats), axis=0)
    assert mats_unique.shape[0] == 24, "Should have 24 unique rotations"
    return list(mats_unique)


ROT_MATS = _generate_rot_mats()


def rotate_thickness_matrix_3d(old_faces: np.ndarray, orient_id: int) -> np.ndarray:
    """
    Parameters
    ----------
    old_faces : np.ndarray, shape (6,)
        Thicknesses in order (Z- Z+ Y- Y+ X- X+).
    orient_id : int 0-23
        Index into ROT_MATS corresponding to block orientation.

    Returns
    -------
    new_faces : np.ndarray, shape (6,)
        Thicknesses arranged for the oriented block in *same* ordering
        (Z- Z+ Y- Y+ X- X+).
    """
    if old_faces.shape != (6,):
        raise ValueError("old_faces must be length-6 array")
    R = ROT_MATS[orient_id]                           # (3,3)
    new_faces = np.zeros(6, dtype=old_faces.dtype)
    for new_idx, n_global in enumerate(FACES_3D):
        # Which old face now aligns with this global normal?
        n_old = R.T @ n_global                       # back-transform
        # find exact match in reference normals
        old_idx = np.where((FACES_3D == n_old).all(axis=1))[0][0]
        new_faces[new_idx] = old_faces[old_idx]
    return new_faces


PERMS, FLIPS = [], []          # list[tuple(3)], list[tuple(3)]
for perm in permutations([0, 1, 2]):
    for flips in product([1, -1], repeat=3):
        P = np.eye(3, dtype=int)[:, perm]
        M = P * flips
        if round(np.linalg.det(M)) == 1:   # proper rotation
            PERMS.append(perm)
            FLIPS.append(flips)

assert len(PERMS) == 24    # 共 24 种


def rotate_voxel_24(block: np.ndarray, orient_id: int) -> np.ndarray:
    """
    Rotate voxel block into one of 24 right-handed orientations.

    Parameters
    ----------
    block      : ndarray (Z, Y, X)
    orient_id  : int 0-23  (same index for thickness rotation)

    Returns
    -------
    ndarray rotated to the requested orientation
    """
    perm = PERMS[orient_id]
    flips = FLIPS[orient_id]

    rot = np.transpose(block, axes=perm)

    for ax, sgn in enumerate(flips):
        if sgn == -1:
            rot = np.flip(rot, axis=ax)
    return rot
