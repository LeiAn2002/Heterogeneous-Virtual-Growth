"""
Batch convert CG1->DG0 via **geometry-based remapping** using meshio's
TimeSeriesReader (works with multi-grid/time-series XDMF written by dolfinx).

- CG1: match DoF coordinates to file node coordinates.
- DG0: match DoF (cell barycenter) to file cell barycenter.
- Then L2-project to DG0 if needed.

Requires: meshio (pip install meshio). Optional: scipy for KDTree fallback.
"""

import h5py
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from pathlib import Path
from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_vector, assemble_matrix

# --- meshio timeseries reader (兼容不同版本的导入) ---
try:
    from meshio.xdmf import TimeSeriesReader  # meshio>=5
except Exception:
    from meshio.xdmf import XdmfTimeSeriesReader as TimeSeriesReader  # 旧版兼容

import meshio

# ----------------- 参数 ----------------- #
DECIMALS = 12         # 坐标量化精度（哈希键）
KD_FALLBACK = True    # 哈希失败时，是否启用 KDTree 最近邻兜底（需 scipy）
KD_RADIUS = 1e-10     # KDTree 容忍半径

try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    KD_FALLBACK = False


def load_field_from_h5(mesh, h5_path, func_name=None):
    with h5py.File(h5_path, "r") as h5:
        fn_group = list(h5["Function"].keys())[0] if func_name is None else func_name
        data = h5[f"Function/{fn_group}/0"][()]

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    is_dg0    = (data.size == num_cells)

    if is_dg0:                                   # reorder by (y,x)
        if mesh.topology.connectivity(mesh.topology.dim, 0) is None:
            mesh.topology.create_connectivity(mesh.topology.dim, 0)
        conn  = mesh.topology.connectivity(mesh.topology.dim, 0)
        cells = np.array([conn.links(i)
                          for i in range(mesh.topology.index_map(mesh.topology.dim).size_local)])
        cent  = mesh.geometry.x[cells].mean(axis=1)
        sort_idx = np.lexsort((cent[:, 0], cent[:, 1]))
        data = data[sort_idx]

    V = fem.functionspace(mesh, ("DG", 0) if is_dg0 else ("CG", 1))
    f = fem.Function(V, name=fn_group)
    loc0, loc1 = V.dofmap.index_map.local_range
    f.vector.array[:] = data[loc0:loc1].flatten()
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                         mode=PETSc.ScatterMode.FORWARD)
    return f, is_dg0, num_cells


# -------- utilities -------- #
def _flatten_1d(a):
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    elif a.ndim != 1:
        raise ValueError(f"Unexpected dataset shape {a.shape}; expected (N,) or (N,1)")
    return a


def key_coord(x):
    return tuple(np.round(np.asarray(x, dtype=float), DECIMALS))


def build_lookup(points):
    pts = np.asarray(points, dtype=float)
    mapping = {key_coord(p): i for i, p in enumerate(pts)}
    tree = cKDTree(pts) if KD_FALLBACK and _HAVE_SCIPY else None
    return mapping, tree, pts


def find_index(x, mapping, tree=None, pts=None):
    k = key_coord(x)
    j = mapping.get(k, None)
    if j is not None:
        return j
    if tree is not None:
        dist, idx = tree.query(np.asarray(x, dtype=float), k=1, distance_upper_bound=KD_RADIUS)
        if np.isfinite(dist) and idx < len(pts):
            return int(idx)
    raise KeyError(f"coordinate {x} not found (tol={KD_RADIUS}, decimals={DECIMALS})")


# meshio cell 类型 -> 维度
_CELL_DIM = {
    # 2D
    "triangle": 2, "triangle6": 2, "triangle10": 2,
    "quad": 2, "quad8": 2, "quad9": 2,
    # 3D
    "tetra": 3, "tetra10": 3,
    "hexahedron": 3, "hexahedron20": 3, "hexahedron27": 3,
    "wedge": 3, "pyramid": 3,
    # 1D（列出来以防万一）
    "line": 1, "line3": 1,
}


def pick_topdim_block(cells, topdim):
    """从 meshio TimeSeriesReader 的 cells 列表选择维度=topdim 的第一个块"""
    for ib, cb in enumerate(cells):
        ctype = cb.type if hasattr(cb, "type") else cb[0]
        if _CELL_DIM.get(ctype, -1) == topdim:
            return ib, ctype
    raise RuntimeError(f"No cell block with topdim={topdim} found in XDMF")


# -------- L2 projector（与你原来一致） -------- #
def l2_project(src, V_dst):
    u, v = ufl.TrialFunction(V_dst), ufl.TestFunction(V_dst)
    A = assemble_matrix(fem.form(ufl.inner(u, v) * ufl.dx)); A.assemble()
    b = assemble_vector(fem.form(ufl.inner(src, v) * ufl.dx))
    dst = fem.Function(V_dst, name=f"{src.name}_DG0")
    ksp = PETSc.KSP().create(V_dst.mesh.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    ksp.solve(b, dst.vector)
    dst.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                           mode=PETSc.ScatterMode.FORWARD)
    return dst


# -------- 通过 TimeSeriesReader + 几何匹配 读取函数 -------- #
def load_field_geom_match_timeseries(mesh, xdmf_path, h5_path, prefer_name=None,
                                     step=0, verbose=True):
    """
    读取 XDMF timeseries，并将 prefer_name 对应的标量场通过“几何坐标匹配”
    映射到当前 mesh 的函数空间。自动对齐文件侧与 DoF 坐标的维度。
    返回 (f_src, is_dg0)
    """
    # 兼容导入
    try:
        from meshio.xdmf import TimeSeriesReader
    except Exception:
        from meshio.xdmf import XdmfTimeSeriesReader as TimeSeriesReader

    comm = mesh.comm
    topdim = mesh.topology.dim  # 仅用于选择 cell block

    # 1) 从 H5 拿函数名
    with h5py.File(h5_path, "r") as h5:
        groups = list(h5["Function"].keys()) if "Function" in h5 else []
        if prefer_name is None:
            if not groups:
                raise RuntimeError("Cannot infer function name; please provide prefer_name.")
            prefer_name = groups[0]

    # 2) 用 TimeSeriesReader 读取 points/cells 与第 step 步数据
    with TimeSeriesReader(str(xdmf_path)) as rdr:
        points_file, cells = rdr.read_points_cells()
        # 选择与拓扑维度一致的那个 cell block
        ib = None
        for k, cb in enumerate(cells):
            ctype = cb.type if hasattr(cb, "type") else cb[0]
            if _CELL_DIM.get(ctype, -1) == topdim:
                ib = k
                break
        if ib is None:
            raise RuntimeError(f"No cell block with topdim={topdim} found in XDMF")
        t, point_data, cell_data = rdr.read_data(step)
        vals_point = point_data.get(prefer_name, None)
        vals_cell_list = cell_data.get(prefer_name, None)
        vals_cell = None if vals_cell_list is None else vals_cell_list[ib]

    # 3) 判定 DG0 / CG1，并取“文件侧坐标”与“文件侧值”
    cells_file = cells[ib].data  # (N_cells, n_verts)
    if vals_cell is not None and vals_point is None:
        is_dg0 = True
        vals_file = _flatten_1d(vals_cell)
        file_coords = np.mean(points_file[cells_file], axis=1)   # 单元重心，形状 (Nc, df)
    elif vals_point is not None and vals_cell is None:
        is_dg0 = False
        vals_file = _flatten_1d(vals_point)
        file_coords = np.asarray(points_file)                    # 节点坐标，形状 (Np, df)
    else:
        raise RuntimeError(
            f"Cannot determine centering for '{prefer_name}': "
            f"point_data={'Y' if vals_point is not None else 'N'}, "
            f"cell_data={'Y' if vals_cell is not None else 'N'}"
        )

    # 4) 目标空间与 DoF 坐标（自适应获取）
    V = fem.functionspace(mesh, ("DG", 0) if is_dg0 else ("CG", 1))
    f = fem.Function(V, name=prefer_name)
    loc0, loc1 = V.dofmap.index_map.local_range
    nloc = loc1 - loc0

    coords = np.asarray(V.tabulate_dof_coordinates())
    if coords.ndim == 1:
        if coords.size % nloc != 0:
            raise ValueError(f"Unexpected dof_coordinates size {coords.size} for nloc={nloc}")
        dof_xyz = coords.reshape(nloc, coords.size // nloc)
    else:
        if coords.shape[0] != nloc:
            if coords.size % nloc != 0:
                raise ValueError(f"Unexpected dof_coordinates shape {coords.shape} for nloc={nloc}")
            dof_xyz = coords.reshape(nloc, coords.size // nloc)
        else:
            dof_xyz = coords

    # 5) **对齐维度**：统一到 d = min(文件侧维度, DoF 维度)
    file_coords = np.asarray(file_coords, dtype=float)
    d = min(file_coords.shape[1], dof_xyz.shape[1])
    file_use = file_coords[:, :d].copy()
    dof_use  = dof_xyz[:,  :d].copy()

    # 6) 构建文件侧查找结构（哈希 + KDTree兜底）
    map_file, tree_file, pts_file = build_lookup(file_use)

    # 7) 逐个本地 DoF 匹配赋值（使用对齐后的 d 维坐标）
    misses = 0
    for i_local in range(nloc):
        try:
            j_file = find_index(dof_use[i_local], map_file, tree_file, pts_file)
            f.vector.array[i_local] = vals_file[j_file]
        except KeyError:
            misses += 1
            f.vector.array[i_local] = 0.0
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                         mode=PETSc.ScatterMode.FORWARD)

    if verbose and comm.rank == 0:
        kind = "DG0(Cell)" if is_dg0 else "CG1(Node)"
        print(f"[info] Loaded '{prefer_name}' as {kind} via geometry matching. "
              f"local_misses={misses}; file_dim={file_coords.shape[1]}, dof_dim={dof_xyz.shape[1]}, used_dim={d}")

    return f, is_dg0



# -------------------- 批处理主程序 -------------------- #
input_files = [
    "./datas/data_maple/rho_field.xdmf",
    "./datas/data_maple/ksi_field_1.xdmf",
    "./datas/data_maple/ksi_field_2.xdmf",
    "./datas/data_maple/ksi_field_3.xdmf",
    "./datas/data_maple/ksi_field_4.xdmf",
    "./datas/data_maple/ksi_field_5.xdmf",
    "./datas/data_maple/vf_field.xdmf",
]
suffix_out = "_DG0.xdmf"

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    for xfile in input_files:
        xfile = Path(xfile)
        h5file = xfile.with_suffix(".h5")
        if not h5file.exists():
            raise FileNotFoundError(f"{h5file} missing.")

        # (1) 读 mesh（dolfinx 原生）
        with io.XDMFFile(comm, str(xfile), "r") as xdmf:
            mesh = xdmf.read_mesh()

        # (2) 通过 TimeSeriesReader + 几何匹配 读取源场
        #     prefer_name 从 H5 的 Function/<name> 拿第一个
        with h5py.File(h5file, "r") as h5:
            groups = list(h5["Function"].keys())
            prefer_name = groups[0] if groups else None

        f_src, already_dg0 = load_field_geom_match_timeseries(
            mesh, str(xfile), str(h5file), prefer_name=prefer_name, step=0, verbose=True
        )

        # (3) 若不是 DG0，则做 L2 投影
        if already_dg0:
            f_dg0 = f_src
        else:
            V_dg0 = fem.functionspace(mesh, ("DG", 0))
            f_dg0 = l2_project(f_src, V_dg0)

        # (4) 写结果
        out = xfile.with_name(xfile.stem + suffix_out.replace(".xdmf", "") + ".xdmf")
        with io.XDMFFile(comm, str(out), "w") as xout:
            xout.write_mesh(mesh)
            xout.write_function(f_dg0, t=0.0)

        if comm.rank == 0:
            print(f"[OK] {xfile.name} -> {out.name}")
