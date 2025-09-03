#!/usr/bin/env python3
# run_virtual_growth_design_order.py
#
# 读取 *_DG0_proj.xdmf 场，使用“TimeSeriesReader + 几何匹配”读入为 fem.Function，
# 然后将 DG0 向量按 (y, x) 稳定排序（设计序），供虚拟生长算法使用。

import time
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI
from dolfinx import io, fem

# ✅ 从你的 project_script.py 中导入几何匹配读取函数
from scripts.project_script import load_field_geom_match_timeseries
# 如果有需要从 CG1 投到 DG0，可同时导入 l2_project：
# from scripts.project_script import load_field_geom_match_timeseries, l2_project

# 你原本的入口
from virtual_growth.main import main

# ---------------- 用户可调参数 ----------------
mesh_number = 100
element_number = 2
mesh_size = (mesh_number, mesh_number)
element_size = (element_number, element_number)

candidates = ["star", "gripper", "T", "V", "O"]
m = 0.75

save_path = "designs/2d/"
fig_name = "symbolic_graph.jpg"
gif_name = "symbolic_graph.gif"

# 这些文件应为“投回正方形参考网格后”的 DG0 场
input_files = [
    "./datas/data_maple/data_after_project/projected_fix/rho_field_DG0_proj.xdmf",
    "./datas/data_maple/data_after_project/projected_fix/ksi_field_1_DG0_proj.xdmf",
    "./datas/data_maple/data_after_project/projected_fix/ksi_field_2_DG0_proj.xdmf",
    "./datas/data_maple/data_after_project/projected_fix/ksi_field_3_DG0_proj.xdmf",
    "./datas/data_maple/data_after_project/projected_fix/ksi_field_4_DG0_proj.xdmf",
    "./datas/data_maple/data_after_project/projected_fix/ksi_field_5_DG0_proj.xdmf",
    "./datas/data_maple/data_after_project/projected_fix/vf_field_DG0_proj.xdmf",
]
# ------------------------------------------------


def _tabulate_dof_coords(V: fem.FunctionSpace) -> np.ndarray:
    """
    兼容不同 dolfinx 版本，返回本 rank 拥有 DOF 的坐标，形状 (nloc, gdim_eff)。
    """
    coords = np.asarray(V.tabulate_dof_coordinates())
    nloc = V.dofmap.index_map.local_range[1] - V.dofmap.index_map.local_range[0]
    if coords.ndim == 1:
        assert coords.size % nloc == 0, f"Unexpected dof coord size {coords.size} for nloc={nloc}"
        return coords.reshape(nloc, coords.size // nloc)
    if coords.shape[0] != nloc:
        assert coords.size % nloc == 0, f"Unexpected dof coord shape {coords.shape} for nloc={nloc}"
        return coords.reshape(nloc, coords.size // nloc)
    return coords


def dg0_function_to_design_array(f: fem.Function, y_desc: bool = False) -> np.ndarray:
    """
    将 DG0 函数向量重排为稳定的“设计序”一维数组：
      - 先按 y 排序（默认升序；y_desc=True 为降序），再按 x 排序。
      - 返回长度 = 全局单元数 的 ndarray（在所有 rank 上一致）。

    说明：
      - 对 DG0，tabulate_dof_coordinates() 返回的就是单元重心坐标，且顺序与 f.x.array 对应。
      - 若几何为 3D（z=0 的片），只取前两列 (x,y) 排序即可。
    """
    V = f.function_space
    mesh = V.mesh
    comm = mesh.comm

    # 本地坐标与本地值（顺序一致）
    xy_local = _tabulate_dof_coords(V)[:, :2].copy()  # 取 x,y
    val_local = np.asarray(f.x.array).copy()

    # 汇总到 rank 0；注意 gather 后顺序是“各 rank 本地顺序拼接”
    xy_list = comm.gather(xy_local, root=0)
    val_list = comm.gather(val_local, root=0)

    if comm.rank == 0:
        XY = np.vstack(xy_list)
        VAL = np.concatenate(val_list)
        # numpy.lexsort 的“最后一个键”为主键
        ykey = -XY[:, 1] if y_desc else XY[:, 1]
        order = np.lexsort((XY[:, 0], ykey))  # 主键：y；次键：x
        arr = VAL[order]
    else:
        arr = None

    # 广播结果，保证所有 rank 拿到一致的一维数组
    arr = comm.bcast(arr, root=0)
    return arr


def _normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """对每行做归一化，避免除零。"""
    s = A.sum(axis=1, keepdims=True)
    s = np.where(s > eps, s, 1.0)
    return A / s


def main_driver():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    num_elems = int(np.prod(mesh_size))
    value_list = []

    # 逐个读取场：XDMF 读网格，H5 找函数名，然后用“几何匹配”取回 fem.Function（DG0）
    for xfile in input_files:
        xfile = Path(xfile)
        h5file = xfile.with_suffix(".h5")
        if not h5file.exists():
            raise FileNotFoundError(f"{h5file} missing.")

        # (1) 读 mesh
        with io.XDMFFile(comm, str(xfile), "r") as xdmf:
            mesh = xdmf.read_mesh()

        # (2) 从 H5 获取函数名
        with h5py.File(h5file, "r") as h5:
            groups = list(h5["Function"].keys()) if "Function" in h5 else []
            if not groups:
                raise RuntimeError(f"No Function group in {h5file}")
            fn_name = groups[0]

        # (3) 几何匹配读取（不依赖 DoF 编号/并行划分）
        f_src, is_dg0 = load_field_geom_match_timeseries(
            mesh, str(xfile), str(h5file), prefer_name=fn_name, step=0, verbose=False
        )
        # 如果极端情况下不是 DG0，可在这里投到 DG0（通常不会触发，因为 *_DG0_proj 是 DG0）
        # if not is_dg0:
        #     V0 = fem.functionspace(mesh, ("DG", 0))
        #     f_src = l2_project(f_src, V0)

        # (4) ✨ 重排为“设计序”一维数组（与 mesh_number×mesh_number 对齐）
        arr_design = dg0_function_to_design_array(f_src, y_desc=False)
        if arr_design.size != num_elems:
            raise ValueError(
                f"设计序数组长度 {arr_design.size} 与 mesh_number^2={num_elems} 不一致。"
                "请确认参考网格划分与 mesh_number 一致。"
            )
        value_list.append(arr_design)

    # ====== 你原脚本的后续逻辑（保持不变，改用数组而非 f.x.array） ======
    # rho_field = value_list[0]
    ksi_field_1 = value_list[1]
    void = np.where(ksi_field_1 > 0.7)[0]  # 或者改阈值

    # 频率提示（按候选块数量拼接后转置）
    frequency_hints_array = np.vstack([value_list[i + 1] for i in range(len(candidates))]).T
    frequency_hints = _normalize_rows(frequency_hints_array)

    v = value_list[-1]
    r_array = np.full((num_elems,), 0.1, dtype=float)
    v_array = np.vstack((v - 0.05, v + 0.05)).T

    if rank == 0:
        print(f"[info] frequency_hints shape: {frequency_hints.shape}")
        print(f"[info] v_array shape: {v_array.shape}, r_array shape: {r_array.shape}")
        print(f"[info] void count: {None if void is None else void.size}")

    # 运行虚拟生长
    start_time = time.time()
    main(mesh_size, element_size, candidates, frequency_hints, v_array, r_array, m, void,
         periodic=True, num_tries=2000, print_frequency=False, make_figure=True,
         make_gif=False, color="#96ADFC", save_path=save_path, fig_name=fig_name,
         gif_name=gif_name,
         save_mesh=True, save_mesh_path=save_path,
         save_mesh_name="symbolic_graph.npy")
    if rank == 0:
        print("Virtual Growth Time: ", time.time() - start_time)


if __name__ == "__main__":
    main_driver()
