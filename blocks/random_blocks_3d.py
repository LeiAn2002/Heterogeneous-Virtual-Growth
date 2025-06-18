"""
3-D building-block generation in voxel space (global-point version)
------------------------------------------------------------------
This revision uses a **global point table** so that multiple curves can share
common control points even after randomisation.

Workflow
~~~~~~~~
1.  Define **all control points** once; mark each as *pinned* (non-randomised)
    or *free* (subject to perturbation).
2.  Provide **curve definitions** as lists of point indices referring to the
    global table and per-curve radius-scaling (vf) arrays.
3.  Randomise only the free points inside a bounding box.
4.  Evaluate B-splines, variable radii, voxelisation, filtering, and Heaviside
    projection as before.

Compared with the original per-curve randomisation, this ensures that shared
points remain coincident across curves (e.g., X/Y/Z axes meeting at origin).
"""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter
from tqdm import tqdm
import pyvista as pv
import random
from typing import List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Helper functions (unchanged)
# ---------------------------------------------------------------------------


def heaviside_projection(rho_tilde, beta: float = 8.0, theta: float = 0.5):
    """Continuous Heaviside projection (Sigmund 2007)."""
    num = np.tanh(beta * theta) + np.tanh(beta * (rho_tilde - theta))
    den = np.tanh(beta * theta) + np.tanh(beta * (1.0 - theta))
    return num / den


def build_discs_from_boundary(ctrl_pts: Sequence[Tuple[float, float, float]],
                              vf_ctrl: Sequence[float],
                              bb_lim: float = 1.0,
                              tol: float = 1e-6):
    """Detect whether endpoints lie on the cube faces and, if so, create a disc."""
    discs = []
    for p, rad in zip((ctrl_pts[0], ctrl_pts[-1]), (vf_ctrl[0], vf_ctrl[-1])):
        axis = int(np.argmax(np.abs(p)))
        sign = 1 if p[axis] > 0 else -1
        # Skip if not on face
        if not np.isclose(np.abs(p[axis]), bb_lim, atol=tol):
            continue
        if axis == 0:
            cu, cv = p[1], p[2]  # X face
        elif axis == 1:
            cu, cv = p[0], p[2]  # Y face
        else:
            cu, cv = p[0], p[1]  # Z face
        discs.append((axis, sign, cu, cv, rad))
    return discs


def dist_to_forbidden_faces(points: np.ndarray,
                            discs,
                            bb_lim: float = 1.0):
    """Compute distance from each point to forbidden regions of cube faces."""
    pts = np.asarray(points)
    M = len(pts)
    min_dist = np.full(M, np.inf)

    # Index discs by face
    disc_dict = {(d[0], d[1]): d for d in discs}

    for axis in range(3):
        for sign in (-1, 1):
            face_coord = sign * bb_lim
            d_norm = np.abs(pts[:, axis] - face_coord)  # normal distance

            disc = disc_dict.get((axis, sign))
            if disc is not None:
                _, _, cu, cv, rad = disc
                if axis == 0:
                    u, v = pts[:, 1], pts[:, 2]
                elif axis == 1:
                    u, v = pts[:, 0], pts[:, 2]
                else:
                    u, v = pts[:, 0], pts[:, 1]
                d_plane = np.hypot(u - cu, v - cv) - rad
                mask_out = d_plane > 0
                d_total = np.hypot(d_norm[mask_out], d_plane[mask_out])
                min_dist[mask_out] = np.minimum(min_dist[mask_out], d_total)
            else:
                min_dist = np.minimum(min_dist, d_norm)
    return min_dist


def modify_radius_with_vf(orig_r: np.ndarray,
                          vf_ctrl: Sequence[float],
                          u_param: np.ndarray):
    """Scale the nearest-distance radius by interpolated vf values."""
    knots = np.linspace(0, 1, len(vf_ctrl))
    vf_line = np.interp(u_param, knots, vf_ctrl)
    return orig_r * vf_line


def generate_bspline(points3d: Sequence[Tuple[float, float, float]],
                     n_samples: int = 200):
    """Return B-spline points and parameter values."""
    pts = np.asarray(points3d)
    m = len(pts)
    tck, _ = splprep(pts.T, s=0.0, k=min(3, m - 1))
    u = np.linspace(0, 1, n_samples)
    x, y, z = splev(u, tck)
    return np.vstack([x, y, z]).T, u


def paint_tube_into_voxel(volume: np.ndarray,
                          tube_pts: np.ndarray,
                          radii: np.ndarray,
                          pitch: float,
                          bb_min: np.ndarray):
    """Voxelise a single tubular curve into the binary volume (in-place)."""
    Nz, Ny, Nx = volume.shape
    for p, r in zip(tube_pts, radii):
        x0, y0, z0 = (p - r - bb_min) / pitch
        x1, y1, z1 = (p + r - bb_min) / pitch
        ix0, iy0, iz0 = np.floor([x0, y0, z0]).astype(int).clip(0, [Nx - 1, Ny - 1, Nz - 1])
        ix1, iy1, iz1 = np.ceil([x1, y1, z1]).astype(int).clip(0, [Nx - 1, Ny - 1, Nz - 1])

        xs = (np.arange(ix0, ix1 + 1) + 0.5) * pitch + bb_min[0]
        ys = (np.arange(iy0, iy1 + 1) + 0.5) * pitch + bb_min[1]
        zs = (np.arange(iz0, iz1 + 1) + 0.5) * pitch + bb_min[2]

        dx = xs[:, None, None] - p[0]
        dy = ys[None, :, None] - p[1]
        dz = zs[None, None, :] - p[2]
        inside = (dx ** 2 + dy ** 2 + dz ** 2) < r ** 2
        volume[iz0:iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1] |= inside.transpose(2, 1, 0).astype(np.uint8)

# ---------------------------------------------------------------------------
# New point-based randomisation utilities
# ---------------------------------------------------------------------------


def randomise_points(points: List[Tuple[float, float, float]],
                     pin_flags: List[bool],
                     radius: float,
                     bb_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
                     bb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                     max_try: int = 100):
    """Randomly perturb **free** points within a sphere of given radius."""
    new_pts = []
    for p, pinned in zip(points, pin_flags):
        if pinned:
            new_pts.append(tuple(p))
            continue
        bx, by, bz = p
        for _ in range(max_try):
            rr = random.uniform(0, radius)
            phi = random.uniform(0, 2 * np.pi)
            cost = random.uniform(-1, 1)
            sint = np.sqrt(1 - cost * cost)
            cx = bx + rr * sint * np.cos(phi)
            cy = by + rr * sint * np.sin(phi)
            cz = bz + rr * cost
            if (bb_min[0] <= cx <= bb_max[0] and
                bb_min[1] <= cy <= bb_max[1] and
                bb_min[2] <= cz <= bb_max[2]):
                new_pts.append((cx, cy, cz))
                break
        else:
            new_pts.append((bx, by, bz))  # fallback
    return new_pts

# ---------------------------------------------------------------------------
# Main high-level function (re-designed API)
# ---------------------------------------------------------------------------


def voxel_building_block(points: List[Tuple[float, float, float]],
                         pin_flags: List[bool],
                         curve_defs: List[List[int]],
                         vf_groups: List[List[float]],
                         random_radius: float = 0.5,
                         pitch: float = 0.02,
                         r_filter_vox: int = 3,
                         beta: float = 128.0,
                         theta: float = 0.5):
    """Generate a binary/filtered voxel model from global points and curves."""

    bb_min = np.array([-1.0, -1.0, -1.0])
    bb_max = np.array([1.0, 1.0, 1.0])

    # 1) randomise free points
    rnd_pts = randomise_points(points, pin_flags, random_radius,
                               bb_min=bb_min, bb_max=bb_max)

    # 2) evaluate each curve
    curves_pts, curves_radii = [], []
    for pts_idx, vf_ctrl in zip(curve_defs, vf_groups):
        ctrl_pts = [rnd_pts[i] for i in pts_idx]
        spline_pts, u = generate_bspline(ctrl_pts, n_samples=300)
        discs = build_discs_from_boundary(ctrl_pts, vf_ctrl, bb_lim=1.0)
        orig_r = dist_to_forbidden_faces(spline_pts, discs, bb_lim=1.0)
        radii = modify_radius_with_vf(orig_r, vf_ctrl, u)
        curves_pts.append(spline_pts)
        curves_radii.append(radii)

    # 3) voxel grid
    Nx, Ny, Nz = np.ceil((bb_max - bb_min) / pitch).astype(int)
    volume = np.zeros((Nz, Ny, Nx), dtype=np.uint8)

    for pts, rad in zip(curves_pts, curves_radii):
        paint_tube_into_voxel(volume, pts, rad, pitch, bb_min)

    # 4) filtering + projection
    if r_filter_vox > 0:
        vol_float = uniform_filter(volume.astype(np.float32),
                                   size=r_filter_vox, mode="constant", cval=0.0)
    else:
        vol_float = volume.astype(np.float32)
    rho_bar = heaviside_projection(vol_float, beta=beta, theta=theta)
    # return rho_bar, volume, (Nx, Ny, Nz), pitch, bb_min, rnd_pts
    return rho_bar

# ---------------------------------------------------------------------------
# Simple visualisation wrapper (unchanged)
# ---------------------------------------------------------------------------


def plot_voxel_solid(volume_or_rho: np.ndarray,
                     pitch: float,
                     bb_min: Tuple[float, float, float],
                     color: str = "skyblue",
                     opacity_value: float = 1.0):
    """Render the voxel model via PyVista."""
    Nz, Ny, Nx = volume_or_rho.shape
    print("voxel dimensions:", Nz, Ny, Nx)
    grid = pv.ImageData(dimensions=(Nx + 1, Ny + 1, Nz + 1),
                        spacing=(pitch, pitch, pitch),
                        origin=bb_min)
    grid.cell_data["rho"] = volume_or_rho.ravel(order="F")
    p = pv.Plotter()
    p.add_volume(grid, scalars="rho",
                 opacity=[0.0, opacity_value], cmap=[color, color],
                 shade=False, preference="cell")
    p.show_grid(color="lightgray")
    p.show(title="Solid voxel block")

# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Global point list
    pts = [
        (-1, 0, 0),  # 0
        (0, 0, 0),   # 1 shared origin
        (1, 0, 0),   # 2
        (0, -1, 0),  # 3
        (0, 1, 0),   # 4
        (0, 0, -1),  # 5
        (0, 0, 1)    # 6
    ]

    # Pinned flags: endpoints fixed, origin free → False
    pins = [True, False, True, True, True, True, True]

    # Curve definitions by point indices
    curves = [[0, 1, 2],   # X-axis curve
              [3, 1, 4],   # Y-axis curve
              [5, 1, 6]]   # Z-axis curve

    vf_groups = [
        [0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3]
    ]

    rho_bar = voxel_building_block(
        points=pts, pin_flags=pins,
        curve_defs=curves, vf_groups=vf_groups,
        random_radius=0.5, pitch=0.04,
        r_filter_vox=0)
    
    # save rho_bar to txt
    np.savetxt("rho_bar.txt", rho_bar.flatten(), fmt='%.6f')

    plot_voxel_solid(rho_bar, pitch=0.04, bb_min=[-1, -1, -1],
                     color="black", opacity_value=1.0)
