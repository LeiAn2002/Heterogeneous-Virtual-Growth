import numpy as np
from itertools import permutations, product
from typing import List, Tuple


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

# ---------- 6 faces order ： Z- Z+ Y- Y+ X- X+ ----------
FACES_3D = np.array([
    [0,  0, -1],   # Z-
    [0,  0,  1],   # Z+
    [0, -1,  0],   # Y-
    [0,  1,  0],   # Y+
    [-1, 0,  0],   # X-
    [1,  0,  0],   # X+
], dtype=int)


# ──────────────────────────────────────────────────────────
# elementary 90-deg rotation matrices
# ──────────────────────────────────────────────────────────
def R_x(k: int): return np.linalg.matrix_power(
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], int), k % 4)


def R_y(k: int): return np.linalg.matrix_power(
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], int), k % 4)


def R_z(k: int): return np.linalg.matrix_power(
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], int), k % 4)


ROT_MATS, PERMS, FLIPS = [], [], []


def _append(R):
    ROT_MATS.append(R)
    # derive perm / flips for rotate_voxel_24
    perm = tuple(np.argmax(np.abs(R), axis=0))         # where old X,Y,Z go
    flips = tuple(int(R[row, col]) for col, row in enumerate(perm))
    PERMS.append(perm)
    FLIPS.append(flips)


# group-A
for k in range(4):
    _append(R_z(k))
# group-B  front→top  (Rx -90°)
for k in range(4):
    _append(R_z(k) @ R_x(-1))
# group-C  front→bottom (Rx +90°)
for k in range(4):
    _append(R_z(k) @ R_x(1))
# group-D  front→left (Ry +90°)
for k in range(4):
    _append(R_z(k) @ R_y(1))
# group-E  front→right (Ry -90°)
for k in range(4):
    _append(R_z(k) @ R_y(-1))
# group-F  front→back  (Rx 180°)
for k in range(4):
    _append(R_z(k) @ R_x(2))

# sanity-check
assert len(ROT_MATS) == 24, "must have 24 distinct right-hand rotations"
assert len({R.tobytes() for R in ROT_MATS}) == 24, "no duplicates!"

# ROT_TO_ID = {R.tobytes(): i for i, R in enumerate(ROT_MATS)}

new2old = [
     0, 12, 22, 16,  4,  7,  6,  5,
     20, 18,  2, 14,  8,  9, 10, 11,
     3, 15, 21, 19,  1, 13, 23, 17
]

ROT_MATS = [ROT_MATS[i] for i in new2old]
PERMS = [PERMS[i] for i in new2old]
FLIPS = [FLIPS[i] for i in new2old]


def rotate_thickness_matrix_3d(faces6: np.ndarray, oid: int) -> np.ndarray:
    """faces6: [Z-,Z+,Y-,Y+,X-,X+]"""
    thick = faces6.reshape(3, 2)
    perm, flips = PERMS[oid], FLIPS[oid]
    thick = thick[list(perm), :]

    for axis, f in enumerate(flips):
        if f == -1:
            thick[axis] = thick[axis, ::-1]

    return thick.ravel()


def rotate_voxel_24(arr: np.ndarray, oid: int) -> np.ndarray:
    perm, flips = PERMS[oid], FLIPS[oid]
    rot = np.transpose(arr, axes=perm)
    for ax, s in enumerate(flips):
        if s == -1:
            rot = np.flip(rot, axis=ax)
    return rot


def inv_id(oid: int) -> int:
    """Return orientation-ID whose matrix equals ROT_MATS[oid].T"""
    R_inv = ROT_MATS[oid].T
    for j, R in enumerate(ROT_MATS):
        if (R == R_inv).all():
            return j
  
    raise ValueError("inverse orientation not found (rotation table incomplete)")
