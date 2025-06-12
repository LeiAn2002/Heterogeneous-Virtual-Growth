"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Ke Liu (liuke@pku.edu.cn)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsor:
- David C. Crawford Faculty Scholar Award from the Department of Civil and
  Environmental Engineering and Grainger College of Engineering at the
  University of Illinois

Citations:
- Jia, Y., Liu, K., Zhang, X.S., 2024. Modulate stress distribution with
  bio-inspired irregular architected materials towards optimal tissue support.
  Nature Communications 15, 4072. https://doi.org/10.1038/s41467-024-47831-2
- Jia, Y., Liu, K., Zhang, X.S., 2024. Topology optimization of irregular
  multiscale structures with tunable responses using a virtual growth rule.
  Computer Methods in Applied Mechanics and Engineering 425, 116864.
  https://doi.org/10.1016/j.cma.2024.116864
"""

import numpy as np
from virtual_growth.main import main
from mpi4py import MPI
from pathlib import Path
import time
from dolfinx import io
from scripts.project_script import load_field_from_h5

mesh_number = 50
element_number = 3
mesh_size = (mesh_number, mesh_number)
# # mesh_size = (1, 1)
element_size = (element_number, element_number)
# candidates = ["cross", "T", "O"]
candidates = ["star", "gripper", "T", "V", "O"]
num_elems = np.prod(mesh_size)
# frequency_hints = np.random.rand(num_elems, len(candidates))
# frequency_hints = frequency_hints / np.sum(frequency_hints, axis=1).reshape(-1, 1)

# first_row_v = np.linspace(0.3, 0.7, mesh_number)
# v_array = np.tile(first_row_v, (mesh_number, 1))
# a = v_array
# b = np.hstack([v_array[:, 1:], v_array[:, -1:]])
# vector_matrix = np.stack((a, b), axis=2)
# v_array = vector_matrix.reshape(-1, 2)

# # first_row_fre = np.linspace(1, 0, 6)
# # base_matrix = np.tile(first_row_fre, (6, 1))
# # base_matrix = base_matrix.T
# # flattened = base_matrix.flatten()
# # second_third_row = (1 - flattened)
# # frequency_hints = np.vstack([flattened, second_third_row])
# # frequency_hints = frequency_hints.T

# first_row_r = np.linspace(1.0, 0, mesh_number)
# r_array = np.tile(first_row_r, (mesh_number, 1))
# r_array = r_array.flatten()

# lower_boundary_v = 0.5
# upper_boundary_v = 0.5
# v_array = np.random.uniform(low=lower_boundary_v, high=upper_boundary_v, size=(mesh_number * mesh_number, 2))

# lower_boundary_r = 0.1
# upper_boundary_r = 0.1
# r_array = np.random.uniform(low=lower_boundary_r, high=upper_boundary_r, size=(mesh_number * mesh_number, ))

# d, m, n = 0.5, 0.75, 0.25

m = 0.75
save_path = "designs/2d/"
fig_name = "symbolic_graph.jpg"
gif_name = "symbolic_graph.gif"
# block_names = ["cross", "T", "O"]

input_files = ["./datas/data_cloak_new/data_after_project/rho_field_DG0.xdmf",
               "./datas/data_cloak_new/data_after_project/ksi_field_1_DG0.xdmf",
               "./datas/data_cloak_new/data_after_project/ksi_field_2_DG0.xdmf",
               "./datas/data_cloak_new/data_after_project/ksi_field_3_DG0.xdmf",
               "./datas/data_cloak_new/data_after_project/ksi_field_4_DG0.xdmf",
               "./datas/data_cloak_new/data_after_project/ksi_field_5_DG0.xdmf",
               "./datas/data_cloak_new/data_after_project/vf_field_DG0.xdmf"]  # list your files here

value_list = []
for xfile in input_files:
    stem = Path(xfile).stem
    h5file = Path(xfile).with_suffix(".h5")
    if not h5file.exists():
        raise FileNotFoundError(f"{h5file} missing.")

    # (1) read mesh
    with io.XDMFFile(MPI.COMM_WORLD, xfile, "r") as xdmf:
        mesh = xdmf.read_mesh()

    # (2) read field via h5py
    f_src, already_dg0, num_cells = load_field_from_h5(mesh, h5file)
    value_list.append(f_src)

rho_field = value_list[0].x.array
void = np.where(rho_field < 0.1)[0]
# void = None
frequency_hints_list = []
for i in range(len(candidates)):
    frequency_hints_list.append(value_list[i + 1].x.array)
frequency_hints_array = np.array(frequency_hints_list)
frequency_hints = frequency_hints_array.transpose()
frequency_hints = frequency_hints / np.sum(frequency_hints, axis=1).reshape(-1, 1)

v = value_list[-1].x.array
# print(v.shape)
r_array = np.random.uniform(low=0.1, high=0.1, size=(num_elems,))
v_array = np.vstack((v-0.05, v+0.05)).T
# print(v_array.shape)


if __name__ == "__main__":
    start_time = time.time()
    main(mesh_size, element_size, candidates, frequency_hints, v_array, r_array, m, void,
         periodic=True, num_tries=40, print_frequency=False, make_figure=True,
         make_gif=False, color="#96ADFC", save_path=save_path, fig_name=fig_name,
         gif_name=gif_name,
         save_mesh=True, save_mesh_path=save_path,
         save_mesh_name="symbolic_graph.npy")
    print("Virtual Growth Time: ", time.time() - start_time)

# rho_field = rho_field.reshape(30, 90)
# # # rho_field = np.argsort(mesh_sequ)
# import matplotlib.pyplot as plt
# plt.imshow(rho_field, origin="lower", cmap="RdBu")
# plt.colorbar()
# plt.title("Sorted Field Data")
# plt.savefig("rho_field.png")