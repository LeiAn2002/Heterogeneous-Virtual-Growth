"""
3-D building-block generation in voxel space
--------------------------------------------
1.  Specify 3-D control points (including outer/pinned points) and curve definitions
2.  Discretize curves using SciPy B-spline
3.  Define radius variation along the curve (linear example)
4.  "Draw" tubular structures with variable radius on a voxel grid to generate a binary volume
5.  Naturally merge multiple curves via voxel-wise OR operation (i.e., Boolean union)
6.  Apply 3-D uniform linear filtering + Heaviside projection
"""


import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter
from tqdm import tqdm
import pyvista as pv
import random


# ---------- tools ----------
def heaviside_projection(rho_tilde, beta=8.0, theta=0.5):
    """Sigmoid-Heaviside (Sigmund 2007)"""
    numerator = np.tanh(beta * theta) + np.tanh(beta * (rho_tilde - theta))
    denominator = np.tanh(beta * theta) + np.tanh(beta * (1.0 - theta))
    return numerator / denominator


def build_discs_from_boundary(ctrl_pts, vf_ctrl, bb_lim=1.0, tol=1e-6):
    """
    ctrl_pts  : list[(x,y,z)]
    vf_ctrl   : list[float]，与 ctrl_pts 同长
    返回 discs 列表，每个元素:
        (axis, sign, cu, cv, radius)   radius=对应控制点 vf 值
    """
    discs = []
    for p, rad in zip([ctrl_pts[0], ctrl_pts[-1]],
                      [vf_ctrl[0],  vf_ctrl[-1]]):
        axis = np.argmax(np.abs(p))
        sign = 1 if p[axis] > 0 else -1
        # 若该端点不在任一面 → 跳过生成圆盘
        if not np.isclose(np.abs(p[axis]), bb_lim, atol=tol):
            continue
        # 取 (u,v) = 面内局部坐标
        if axis == 0:   cu, cv = p[1], p[2]      # X 面
        elif axis == 1: cu, cv = p[0], p[2]      # Y 面
        else:           cu, cv = p[0], p[1]      # Z 面
        discs.append((axis, sign, cu, cv, rad))
    return discs


def dist_to_forbidden_faces(points, discs, bb_lim=1.0):
    """
    points : (M,3)
    discs  : 圆盘列表（本曲线独有的开口）
    bb_lim : 立方体半边长，默认 ±1
    """
    pts = np.asarray(points)
    M   = len(pts)
    min_dist = np.full(M, np.inf)

    # 把 discs 按面索引起来，方便 lookup
    disc_dict = {(d[0], d[1]): d for d in discs}

    # 遍历 6 张面
    for axis in range(3):
        for sign in (-1, 1):
            face_coord = sign * bb_lim
            p_axis = pts[:, axis]
            d_norm = np.abs(p_axis - face_coord)        # normal 距离 (M,)

            # 当前面是否有开口圆盘
            disc = disc_dict.get((axis, sign), None)

            # 取局部 (u,v) 坐标
            if axis == 0:
                u, v = pts[:, 1], pts[:, 2]
            elif axis == 1:
                u, v = pts[:, 0], pts[:, 2]
            else:
                u, v = pts[:, 0], pts[:, 1]

            if disc is not None:        # 该面有圆盘，计算到圆盘外缘的距离
                _, _, cu, cv, rad = disc
                d_plane = np.hypot(u - cu, v - cv) - rad       # (M,)
                mask_outside = d_plane > 0                    # 圆盘外
                d_total = np.hypot(d_norm[mask_outside], d_plane[mask_outside])
                min_dist[mask_outside] = np.minimum(min_dist[mask_outside], d_total)
            else:                     # 整张面 forbidden，只需 normal 距离
                min_dist = np.minimum(min_dist, d_norm)

    return min_dist


def modify_radius_with_vf(original_r, vf_ctrl, u_param):
    """
    original_r : (M,) 最近距离数组
    vf_ctrl    : len = 控制点数，内部点 = 放大系数
    u_param    : 样条参数 0~1
    """
    knots   = np.linspace(0, 1, len(vf_ctrl))
    vf_line = np.interp(u_param, knots, vf_ctrl)       # 插值 vf 曲线
    return original_r * vf_line


def randomize_middle_point_3d(ctrl_pts, r,
                              bb_min=(-1.0, -1.0, -1.0),
                              bb_max=( 1.0,  1.0,  1.0),
                              max_try=100):
    """
    ctrl_pts : list/tuple 长度=3 ，[(x0,y0,z0), (x1,y1,z1), (x2,y2,z2)]
    r        : 扰动球半径
    bb_min   : 立方体最小坐标
    bb_max   : 立方体最大坐标
    返回新的 3 个控制点（端点不变，中点随机）
    """
    # 端点保持不变
    p0 = ctrl_pts[0]
    p2 = ctrl_pts[2]

    # 中点随机扰动
    bx, by, bz = ctrl_pts[1]
    for _ in range(max_try):           # 拒绝采样
        rr  = random.uniform(0, r)
        phi = random.uniform(0, 2*np.pi)
        cost = random.uniform(-1, 1)
        sint = np.sqrt(1 - cost*cost)
        cx = bx + rr * sint * np.cos(phi)
        cy = by + rr * sint * np.sin(phi)
        cz = bz + rr * cost
        if (bb_min[0] <= cx <= bb_max[0] and
            bb_min[1] <= cy <= bb_max[1] and
            bb_min[2] <= cz <= bb_max[2]):
            break
    else:
        # 若多次采样仍失败，就保持原中点
        cx, cy, cz = bx, by, bz

    return [p0, (cx, cy, cz), p2]


def generate_bspline(points3d, n_samples=200):
    """
    Given control points Nx3, return (M,3) array uniformly sampled along the curve
    """
    pts = np.asarray(points3d)
    m = len(pts)
    tck, _ = splprep(pts.T, s=0.0, k=min(3, m - 1))
    u = np.linspace(0, 1, n_samples)
    x, y, z = splev(u, tck)
    return np.vstack([x, y, z]).T, u  # (M,3), parameter u∈[0,1]


def paint_tube_into_voxel(volume, tube_pts, radii, pitch, bb_min):
    """
    Discretize a curve into a series of spheres/discs and write to the voxel volume (in-place OR)
    volume  : (Nz, Ny, Nx) 0/1 uint8
    tube_pts: (M, 3)  Sampled points along the curve
    radii   : (M,)    Radius at each point (in world coordinates)
    pitch   : Side length of a single voxel
    bb_min  : Bounding box minimum (x, y, z), used for coordinate → voxel index mapping
    """
    Nz, Ny, Nx = volume.shape
    for p, r in zip(tube_pts, radii):
        # find the local voxel indices
        x0, y0, z0 = (p - r - bb_min) / pitch
        x1, y1, z1 = (p + r - bb_min) / pitch
        ix0, iy0, iz0 = np.floor([x0, y0, z0]).astype(int).clip(0, [Nx-1, Ny-1, Nz-1])
        ix1, iy1, iz1 = np.ceil ([x1, y1, z1]).astype(int).clip(0, [Nx-1, Ny-1, Nz-1])

        # local voxel indices
        xs = (np.arange(ix0, ix1+1) + 0.5)*pitch + bb_min[0]
        ys = (np.arange(iy0, iy1+1) + 0.5)*pitch + bb_min[1]
        zs = (np.arange(iz0, iz1+1) + 0.5)*pitch + bb_min[2]
        dx = xs[:, None, None] - p[0]
        dy = ys[None, :, None] - p[1]
        dz = zs[None, None, :] - p[2]
        dist2 = dx**2 + dy**2 + dz**2
        inside = (dist2 < r**2).transpose(2, 1, 0)
        volume[iz0:iz1+1, iy0:iy1+1, ix0:ix1+1] |= inside.astype(np.uint8)


# ---------- main process ----------
def voxel_building_block(control_points_groups,
                         vf_groups,
                         random_radius=0.5,
                         pitch=0.02,
                         r_filter_vox=3,
                         beta=128, theta=0.5):
    """return rho_bar, volume etc."""

    curves_pts = []   # control points sampled along the curves (Mi, 3)
    curves_radii = []   # Radius at each point (Mi,)

    bb_min = np.array([-1.0, -1.0, -1.0])
    bb_max = np.array([ 1.0,  1.0,  1.0])

    # 1) generate spline curves from control points
    for ctrl_pts, vf_ctrl in zip(control_points_groups, vf_groups):
        # (可选) inner 随机扰动后再使用
        ctrl_pts = randomize_middle_point_3d(ctrl_pts, random_radius)

        pts_curve, u = generate_bspline(ctrl_pts, n_samples=300)

        # --- build opening discs only for boundary endpoints ---
        discs = build_discs_from_boundary(ctrl_pts, vf_ctrl, bb_lim=1.0)

        # --- 原始半径 = 点到 forbidden faces 最近距离 ---
        orig_r = dist_to_forbidden_faces(pts_curve, discs, bb_lim=1.0)

        # --- 最终半径 = orig_r * vf_line (内部插值) ---
        r_curve = modify_radius_with_vf(orig_r, vf_ctrl, u)

        curves_pts.append(pts_curve)
        curves_radii.append(r_curve)

    # 2) calculate bounding box and voxel grid size
    # all_pts = np.vstack(curves_pts)
    # bb_min = all_pts.min(axis=0) - 0.1
    # bb_max = all_pts.max(axis=0) + 0.1

    Nx, Ny, Nz = np.ceil((bb_max - bb_min) / pitch).astype(int)
    volume = np.zeros((Nz, Ny, Nx), dtype=np.uint8)

    # 3) write to voxel (actually traverse each curve)
    for pts_curve, r_curve in tqdm(list(zip(curves_pts, curves_radii)),
                                   desc="Painting tubes"):
        paint_tube_into_voxel(volume, pts_curve, r_curve, pitch, bb_min)

    # 4) optional filtering
    if r_filter_vox > 0:
        vol_float = uniform_filter(volume.astype(np.float32),
                                   size=r_filter_vox, mode='constant', cval=0.0)
    else:
        vol_float = volume.astype(np.float32)

    rho_bar = heaviside_projection(vol_float, beta=beta, theta=theta)
    # rho_bar = vol_float.copy()
    return rho_bar, volume, (Nx, Ny, Nz), pitch, bb_min


def plot_voxel_solid(volume, pitch, bb_min,
                     color="skyblue",
                     opacity_value=1.0):
    """
    volume : (Nz,Ny,Nx) uint8 or bool, only 0/1
    pitch  : voxel size
    bb_min : origin (x0,y0,z0)
    """
    Nz, Ny, Nx = volume.shape

    grid = pv.ImageData(
        dimensions=(Nx+1, Ny+1, Nz+1),
        spacing=(pitch, pitch, pitch),
        origin=bb_min,
    )
    grid.cell_data["rho"] = volume.ravel(order="F")   # still 0/1

    # -- 2. color transfer function --
    opacity_tf = [0.0, opacity_value]   # len==2 => linear projection 0→α0, 1→α1
    cmap_tf = [color, color]         # single color, no interpolation

    # -- 3. rendering --
    p = pv.Plotter()
    p.add_volume(
        grid,
        scalars="rho",
        opacity=opacity_tf,
        cmap=cmap_tf,
        shade=False,               # voxel shell is clear enough, can turn off shading
        preference="cell"          # <<< key: force use cell data → no interpolation
    )
    # p.add_bounds_axes(color="black")
    p.show_grid(color="lightgray")
    p.show(title="Solid voxel block")


if __name__ == "__main__":
    ctrl_x = [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]  # X 轴
    ctrl_y = [(0, -1, 0), (0, 0, 0), (0, 1, 0)]   # Y 轴
    ctrl_z = [(0, 0, -1), (0, 0, 0), (0, 0, 1)]   # Z 轴
    control_groups = [ctrl_x, ctrl_y, ctrl_z]

    vf_groups = [
        [0.25, 0.3, 0.15],   # X 曲线：左端圆盘 0.25, 中点放大 0.6, 右端圆盘 0.15
        [0.5,  0.6, 0.6],    # Y 曲线：起点圆盘 0.3, 其余内部
        [0.3,  0.3, 0.3]     # Z 曲线：无圆盘，全部内部
    ]

    rho_bar, vol_bin, *_ = voxel_building_block(
        control_points_groups=control_groups,
        vf_groups=vf_groups,
        random_radius=0.5,
        pitch=0.04, r_filter_vox=0)
    # save rho_bar to a txt file
    # np.savetxt("rho_bar.txt", rho_bar.flatten(), fmt="%.6f")

    # rho_bin = vol_bin.astype(float)
    plot_voxel_solid(rho_bar, pitch=0.04, bb_min=[-1, -1, -1],
                     color="black", opacity_value=1.0)
    
