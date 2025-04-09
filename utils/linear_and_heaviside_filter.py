from scipy.ndimage import convolve
import numpy as np

# def build_filter_matrix(H, W, r):
#     """
#     Build a sparse matrix for a 2D 'linear filter' of radius r on a (H x W) grid.
#     Each row e corresponds to one pixel (y, x).
#     Each column i also corresponds to one pixel.
#     The entry (e, i) = max(0, r - dist(e,i)) if dist(e,i) < r, else 0.

#     Returns:
#         H_sp: sparse matrix of shape [N, N], N = H*W
#         row_sum: an array of row sums (for normalizing).
#     """
#     # We'll flatten pixel index as e = y*W + x
#     # Helper function: 2D->1D
#     def idx(x, y):
#         return y*W + x

#     row_idx = []
#     col_idx = []
#     data = []

#     r_int = int(np.ceil(r))
#     for y in range(H):
#         for x in range(W):
#             e = idx(x, y)

#             # local window in [x-r_int, x+r_int] etc
#             x_min = max(0, x - r_int)
#             x_max = min(W-1, x + r_int)
#             y_min = max(0, y - r_int)
#             y_max = min(H-1, y + r_int)

#             for yy in range(y_min, y_max+1):
#                 for xx in range(x_min, x_max+1):
#                     dist = np.sqrt((xx - x)**2 + (yy - y)**2)
#                     if dist < r:
#                         w = r - dist
#                         col = idx(xx, yy)
#                         row_idx.append(e)
#                         col_idx.append(col)
#                         data.append(w)

#     N = H * W
#     H_coo = coo_matrix((data, (row_idx, col_idx)), shape=(N, N), dtype=np.float32)
#     H_sp = H_coo.tocsr()
#     row_sum = np.array(H_sp.sum(axis=1)).ravel()  # shape=(N,)
#     return H_sp, row_sum


def build_linear_kernel(r):
    r_int = int(np.ceil(r))
    size = 2 * r_int + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    for dy in range(-r_int, r_int + 1):
        for dx in range(-r_int, r_int + 1):
            dist = np.sqrt(dx**2 + dy**2)
            if dist < r:
                kernel[dy + r_int, dx + r_int] = (r - dist)
    return kernel


def linear_filter(mask, r, mode='constant'):
    """
    Apply a radius-r linear filter to the mask using the sparse matrix approach.
    If H_sp and row_sum are not given, we'll build them on the fly (slower).

    mask: 2D numpy array (H x W)
    r:    filter radius (float)
    H_sp, row_sum: optional precomputed filter matrix and row sums.

    Returns:
        new_mask: 2D array of same shape, filtered by the linear kernel.
    """
    kernel = build_linear_kernel(r)
    
    mask_filtered = convolve(mask, kernel, mode=mode, cval=0.0)
    normalizer = convolve(np.ones_like(mask, dtype=np.float32),
                          kernel, mode=mode, cval=0.0)

    new_mask = np.zeros_like(mask_filtered, dtype=np.float32)
    valid = (normalizer > 1e-12)
    new_mask[valid] = mask_filtered[valid] / normalizer[valid]

    return new_mask


def heaviside(mask, beta=32):
    """
    Apply a Heaviside function to the mask.
    The Heaviside function is approximated by a sigmoid with slope beta.
    """
    return (np.tanh(beta * 0.5) + np.tanh(beta * (mask - 0.5))) / (2 * np.tanh(beta * 0.5))
