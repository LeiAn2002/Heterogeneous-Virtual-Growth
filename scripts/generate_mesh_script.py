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

# import numpy as np
from blocks.block_mesh_2d import generate_mesh
import time


start_time = time.time()
dim = 2
match dim:
    case 2:
        design_path = "./designs/2d/symbolic_graph.npy"  # your input
        geo_file = "./designs/2d/mesh.geo"
        msh_file = "./designs/2d/mesh.msh"
        generate_mesh(design_path, geo_file, msh_file)
        print("Mesh generation time: ", time.time() - start_time)

    # case 3:
    #     m, n = 1.2, 0.0
    #     block_size = 2 * m
    #     data_path = "virtual_growth_data/3d/"
    #     symbolic_graph = np.load("designs/3d/symbolic_graph.npy")
    #     generate_fem_mesh_3d(
    #         symbolic_graph, block_size, data_path,
    #         mesh_path="designs/3d/", check_with_pyvista=True)
