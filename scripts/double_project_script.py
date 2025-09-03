#!/usr/bin/env python3
# project_to_full_mesh_fixed.py
# 把 DG0 场从“带孔网格”投影回完整正方形参考网格（稳健：几何重心映射）

from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.mesh import create_rectangle, CellType, compute_midpoints

# --- meshio timeseries 兼容导入 ---
try:
    from meshio.xdmf import TimeSeriesReader  # meshio >= 5
except Exception:
    from meshio.xdmf import XdmfTimeSeriesReader as TimeSeriesReader  # 旧版

import meshio  # 仅用于类型/常量

# ---------------- 用户参数 ----------------
L, nel = 20.0, 100
input_files = [
    "./datas/data_maple/data_after_project/rho_field_DG0.xdmf",
    "./datas/data_maple/data_after_project/ksi_field_1_DG0.xdmf",
    "./datas/data_maple/data_after_project/ksi_field_2_DG0.xdmf",
    "./datas/data_maple/data_after_project/ksi_field_3_DG0.xdmf",
    "./datas/data_maple/data_after_project/ksi_field_4_DG0.xdmf",
    "./datas/data_maple/data_after_project/ksi_field_5_DG0.xdmf",
    "./datas/data_maple/data_after_project/vf_field_DG0.xdmf",
]
# 孔区默认值（与 input_files 一一对应）
hole_defaults = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

out_dir = Path("./datas/data_maple/data_after_project/projected_fix")
out_dir.mkdir(parents=True, exist_ok=True)

comm = MPI.COMM_WORLD
rank = comm.rank
# -----------------------------------------

# 小工具：meshio cell 类型 -> 维度
_CELL_DIM = {
    "triangle": 2, "triangle6": 2, "triangle10": 2,
    "quad": 2, "quad8": 2, "quad9": 2,
    "tetra": 3, "tetra10": 3,
    "hexahedron": 3, "hexahedron20": 3, "hexahedron27": 3,
    "wedge": 3, "pyramid": 3,
    "line": 1, "line3": 1,
}

def pick_topdim_block(cells, topdim):
    """从 meshio 的 cells 列表里挑出拓扑维度==topdim 的第一个块的下标"""
    for ib, cb in enumerate(cells):
        ctype = cb.type if hasattr(cb, "type") else cb[0]
        if _CELL_DIM.get(ctype, -1) == topdim:
            return ib
    raise RuntimeError(f"No cell block with topdim={topdim} found in XDMF")

# 统一坐标为二维键（与参考网格对齐）
def key2d(pt, decimals=8):
    # 只取 x,y；四舍五入到 1e-8，与你原脚本保持一致
    return (round(float(pt[0]), decimals), round(float(pt[1]), decimals))

# 1) 读取带孔网格（仅用于拿拓扑维度；真正的数据从 XDMF/H5 读）
with io.XDMFFile(comm, input_files[0], "r") as X:
    mesh_hole = X.read_mesh()

cell_dim = mesh_hole.topology.dim  # 2
# 2) 构造无孔参考网格，并预计算参考单元重心键
mesh_ref = create_rectangle(comm, [[-L/2, 0.0], [L/2, L]], [nel, nel], CellType.quadrilateral)
V0_ref = fem.FunctionSpace(mesh_ref, ("DG", 0))

cells_ref = np.arange(mesh_ref.topology.index_map(cell_dim).size_local, dtype=np.int32)
mids_ref = compute_midpoints(mesh_ref, cell_dim, cells_ref)  # (nloc, 2)
keys_ref = [key2d(pt) for pt in mids_ref]

# 3) 遍历所有场
for idx, xdmf_path in enumerate(input_files):
    xdmf_path = Path(xdmf_path)
    h5_path = xdmf_path.with_suffix(".h5")
    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} missing.")

    # --- 从 XDMF timeseries 读取文件侧几何与 cell_data ---
    # 确定函数名（取 H5 里 Function 下的第一个组名）
    import h5py
    with h5py.File(h5_path, "r") as h5:
        if "Function" not in h5 or not h5["Function"].keys():
            raise RuntimeError(f"No Function group in {h5_path}")
        fn_name = list(h5["Function"].keys())[0]

    # 用 TimeSeriesReader 读取 points/cells + 第0步的数据
    with TimeSeriesReader(str(xdmf_path)) as rdr:
        points_file, cells = rdr.read_points_cells()
        ib = pick_topdim_block(cells, cell_dim)
        t, point_data, cell_data = rdr.read_data(0)
        # 对 DG0，我们期望数值在 cell_data 中，并按 cell block 拆分
        vals_cell_list = cell_data.get(fn_name, None)
        if vals_cell_list is None:
            raise RuntimeError(f"'{fn_name}' not found in cell_data of {xdmf_path.name}")
        vals_file = np.asarray(vals_cell_list[ib]).reshape(-1)  # (Nc,)

        # 文件侧单元重心（几何可为3D，这里只取前两维）
        cells_file = cells[ib].data  # (Nc, nverts)
        centroids = np.mean(points_file[cells_file], axis=1)  # (Nc, gdim_file)
        centroids_2d = centroids[:, :2]  # 与参考网格二维坐标对齐

    # --- 建立 “重心坐标键 -> 数值” 字典（全 rank 相同，无需通信）---
    value_dict = {key2d(centroids_2d[i]): float(vals_file[i]) for i in range(vals_file.size)}

    # --- 投影到参考网格：按键查值，孔区用默认值 ---
    f_ref = fem.Function(V0_ref, name=f"{xdmf_path.stem}_proj")
    arr = f_ref.x.array
    default = float(hole_defaults[idx])
    for lid, kpt in enumerate(keys_ref):
        arr[lid] = value_dict.get(kpt, default)
    f_ref.x.scatter_forward()

    # --- 写 XDMF ---
    out_xdmf = out_dir / f"{xdmf_path.stem}_proj.xdmf"
    with io.XDMFFile(comm, str(out_xdmf), "w") as X:
        X.write_mesh(mesh_ref)
        X.write_function(f_ref)

    if rank == 0:
        print(f"✅  {xdmf_path.stem} 已修复并投影 → {out_xdmf.name}")

if rank == 0:
    print("\n🎉  全部场已正确投影（孔区默认值已设置）。")
