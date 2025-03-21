import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.measure import regionprops
import time


def boundary_condition_satisfied(f_bin, margin=0.2):
    """
    Check if the boundary condition is satisfied:
      - The "middle margin fraction" of each edge = 1
      - The remaining portion of that edge = 0
    
    f_bin : 2D binary array, shape (N, N)
    margin : float in (0,1), the fraction of the 'middle' portion 
             of each edge that should be 1
    
    Returns True if the condition is satisfied, False otherwise.
    """
    N, M = f_bin.shape
    assert N == M, "For simplicity, assume square image."
    
    # Middle 20% => from 0.4*N to 0.6*N
    # We'll define the start and end indices
    start_idx = int((1 - margin) / 2 * N)  # e.g. 0.4*N if margin=0.2
    end_idx   = int((1 + margin) / 2 * N)  # e.g. 0.6*N

    # --- Check top edge (row=0) ---
    top_edge = f_bin[0, :]  # shape (N,)
    # part that should be 1
    top_middle = top_edge[start_idx:end_idx]
    # part that should be 0
    top_left   = top_edge[:start_idx]
    top_right  = top_edge[end_idx:]
    
    if not (np.all(top_middle == 1) and np.all(top_left == 0) and np.all(top_right == 0)):
        return False

    # --- Check bottom edge (row=N-1) ---
    bottom_edge = f_bin[N-1, :]
    bottom_middle = bottom_edge[start_idx:end_idx]
    bottom_left   = bottom_edge[:start_idx]
    bottom_right  = bottom_edge[end_idx:]
    
    if not (np.all(bottom_middle == 1) and np.all(bottom_left == 0) and np.all(bottom_right == 0)):
        return False

    # --- Check left edge (col=0) ---
    left_edge = f_bin[:, 0]  # shape (N,)
    left_middle = left_edge[start_idx:end_idx]
    left_top    = left_edge[:start_idx]
    left_bottom = left_edge[end_idx:]
    
    if not (np.all(left_middle == 1) and np.all(left_top == 0) and np.all(left_bottom == 0)):
        return False

    # --- Check right edge (col=N-1) ---
    right_edge = f_bin[:, N-1]
    right_middle = right_edge[start_idx:end_idx]
    right_top    = right_edge[:start_idx]
    right_bottom = right_edge[end_idx:]
    
    if not (np.all(right_middle == 1) and np.all(right_top == 0) and np.all(right_bottom == 0)):
        return False

    # If all edges match the pattern, return True
    return True


def check_connectivity(f_bin):
    """
    Check if the four-side connected with at least 10% pixel-coverage
    """
    top_edge = f_bin[0, :]
    bottom_edge = f_bin[N-1, :]
    left_edge = f_bin[:, 0]
    right_edge = f_bin[:, N-1]
    edges = [top_edge, bottom_edge, left_edge, right_edge]
    for edge in edges:
        if edge.mean() < 0.2:
            return False
        if edge.mean() > 0.5:
            return False
    return True


def check_inner_connectivity(f_bin):
    labeled = label(f_bin, connectivity=1)
    
    # 'labeled.max()' gives the number of distinct connected components.
    num_components = labeled.max()
    
    # If there's exactly one connected component => True
    # If 0, it means there's no white pixel, or if >1 => multiple components
    return (num_components == 1)


def generate_random_field(N=128, alpha=3, tmax=0.2, coverage_thresh=0.1, max_iter=50, seed=None, boundary_margin=0.2):
    """
    Generate a random field via frequency-domain sampling + inverse FFT, 
    followed by binarization and mirroring.
    
    Parameters
    ----------
    N : int
        The size of the grid in both x and y directions (N x N).
    alpha : float
        The exponent in the power-law decay, used in 
        P(k1, k2) ~ (k1^2 + k2^2)^(-alpha/2).
    tmax : float
        The maximum threshold value, from which the actual threshold is 
        uniformly sampled in [0, tmax].
    coverage_thresh : float
        The minimal coverage ratio of the “white” pixels after binarization. 
        For instance, if binarization yields <10% white area, a new sample 
        is generated.
    max_iter : int
        Maximum number of iterations allowed to avoid an infinite loop 
        in extreme situations.
    seed : int or None
        Random seed for reproducibility. If None, no fixed seed is set.
    
    Returns
    ----------
    f_mirror : 2D ndarray
        The final mirrored (2N x 2N) 2D array with values in {0,1}.
    f_bin : 2D ndarray
        The binarized (N x N) result.
    f_real : 2D ndarray
        The real part of the random field (N x N) before standardization.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate the frequency grids k1, k2, used to compute the spectral weight P(k1, k2)
    # Let the frequency index range be -N/2 ~ N/2 - 1, consistent with FFT
    k1 = np.fft.fftfreq(N) * N  
    k2 = np.fft.fftfreq(N) * N
    K1, K2 = np.meshgrid(k1, k2, indexing='ij')  # shape = (N, N)
    
    # Compute (k1^2 + k2^2) while avoiding division by zero
    radius_sq = K1**2 + K2**2
    # Power-law spectrum: P(k1, k2) ~ (k1^2 + k2^2)^(-alpha/2). Here alpha=3.
    # Use a small eps to prevent zero division
    eps = 1e-12
    P = (radius_sq + eps) ** (-alpha/2)
    P[0, 0] = 0.0

    for iteration in range(max_iter):
        # 1) Generate complex noise Z(k1, k2) = X + iY, where X, Y ~ N(0,1)
        X = np.random.normal(loc=0.0, scale=1.0, size=(N, N))
        Y = np.random.normal(loc=0.0, scale=1.0, size=(N, N))
        Z = X + 1j * Y  # complex random matrix     

        # 2) Multiply by the spectral weight: F_k1,k2 = Z_k1,k2 * P(k1, k2)
        F = Z * P

        # 3) Perform 2D inverse FFT on F to obtain the spatial-domain field f(x1, x2)
        # np.fft.ifft2 matches fftfreq indexing automatically
        f_complex = np.fft.ifft2(F)
        
        # 4) Take the real part as our random field
        f_real = np.real(f_complex)

        # 5) Standardize: set the mean to 0 and standard deviation to 1 over all pixels
        mean_val = f_real.mean()
        std_val = f_real.std()
        if std_val < 1e-14:
            # If the standard deviation is too small, the field is almost constant, retry
            continue
        f_std = (f_real - mean_val) / std_val

        # 6) Sample a threshold t ~ Uniform(0, tmax) and binarize f_std
        t = np.random.uniform(-1, 1)
        f_bin = np.zeros_like(f_std, dtype=np.uint8)
        f_bin[f_std > t] = 1  # set pixels above threshold to 1

        # 7) Check coverage & connectivity
        coverage = f_bin.mean()  # proportion of 1s
        if coverage < coverage_thresh:
            # If coverage is too low, sample again
            continue

        # Use skimage's label function to check 4-connectivity
        if not check_connectivity(f_bin):
            continue

        # if not boundary_condition_satisfied(f_bin, margin=boundary_margin):
        #     continue

        if not check_inner_connectivity(f_bin):
            continue
        # By default we only check coverage.
        # If we need exactly one connected component, we might check num_components == 1.
        # if np.abs(coverage-coverage_thresh) > 0.05:
        #     # If conditions are met, mirror the field and return
        #     continue
        if coverage < coverage_thresh:
            continue
        # if not boundary_condition_satisfied(f_bin, margin=boundary_margin):
        #     continue

        f_mirror = mirror_field(f_bin)
        return f_mirror, f_bin, f_real

    # If we fail to generate a valid structure within max_iter, return None
    print(f"Failed to generate valid structure after {max_iter} iterations.")
    return None, None, None


def mirror_field(field_2d):
    """
    Mirror the 2D binary field in the vertical and horizontal directions,
    resulting in a (2N x 2N) image. This can be used to simulate periodic 
    boundaries or other tiling effects.
    """
    top_bottom = np.concatenate([field_2d, np.flipud(field_2d)], axis=0)
    mirrored = np.concatenate([top_bottom, np.fliplr(top_bottom)], axis=1)
    return mirrored


# ============== Test Example ==============
if __name__ == "__main__":
    N = 256    # Grid size
    alpha = 3   # Power-law exponent
    tmax = 1  # Max threshold value
    seed = 72   # Random seed for reproducibility
    boundary_margin = coverage_thresh = 0.1  # Fraction of the edge that should be 1
    start_time = time.time()

    f_mirror, f_bin, f_real = generate_random_field(N=N, alpha=alpha, tmax=tmax, 
                                                    coverage_thresh=coverage_thresh, max_iter=1000000, 
                                                    seed=seed, boundary_margin=boundary_margin)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    if f_mirror is not None:
        # Visualization
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        # The real part of the random field
        axs[0].imshow(f_real, cmap='jet')
        axs[0].set_title("Real part of Random Field")
        axs[0].axis("off")
        
        # Binarized field
        axs[1].imshow(f_bin, cmap='gray')
        axs[1].set_title("Binarized Field")
        axs[1].axis("off")
        
        # Mirrored result
        axs[2].imshow(f_mirror, cmap='gray')
        axs[2].set_title("Mirrored (2N x 2N)")
        axs[2].axis("off")
        
        plt.tight_layout()
        plt.savefig("random_field_example.png", dpi=200)
