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
import time

dim = 2
match dim:
    case 2:
        mesh_size = (6, 6)
        element_size = (4, 4)
        candidates = ["cross", "T"]
        num_elems = np.prod(mesh_size)
        frequency_hints = np.random.rand(num_elems, len(candidates))
        frequency_hints = frequency_hints / np.sum(frequency_hints, axis=1).reshape(-1, 1)
        # v_array = np.array([0.2, 0.35, 0.45, 0.6])
        # r_array = np.array([0.0, 0.0, 0.0, 0.0])
        # v_array = np.array([0.2, 0.2, 0.2, 0.2])

        first_row_v = np.linspace(0.25, 0.75, 6)
        v_array = np.tile(first_row_v, (6, 1))
        a = v_array
        b = np.hstack([v_array[:, 1:], v_array[:, -1:]])
        vector_matrix = np.stack((a, b), axis=2)
        v_array = vector_matrix.reshape(-1, 2)

        # first_row_fre = np.linspace(1, 0, 6)
        # base_matrix = np.tile(first_row_fre, (6, 1))
        # base_matrix = base_matrix.T
        # flattened = base_matrix.flatten()
        # second_third_row = (1 - flattened)
        # frequency_hints = np.vstack([flattened, second_third_row])
        # frequency_hints = frequency_hints.T

        first_row_4 = np.linspace(0.7, 0, 6)
        r_array = np.tile(first_row_4, (6, 1))
        r_array = r_array.flatten()

        # d, m, n = 0.5, 0.75, 0.25
        m = 0.75
        save_path = "designs/2d/"
        fig_name = "symbolic_graph.jpg"
        gif_name = "symbolic_graph.gif"
        block_names = ["cross", "T"]

    case 3:
        mesh_size = (1, 1, 1)
        element_size = (3, 3, 3)
        candidates = ["cross", "t_line", "cross_line"]
        num_elems = np.prod(mesh_size)
        frequency_hints = np.random.rand(num_elems, len(candidates))
        frequency_hints = frequency_hints / np.sum(frequency_hints, axis=1).reshape(-1, 1)
        d, m, n = 0.2, 1.0, 0.0
        data_path = "virtual_growth_data/3d/"
        save_path = "designs/3d/"
        fig_name = "symbolic_graph.jpg"
        gif_name = "symbolic_graph.gif"

if __name__ == "__main__":
    start_time = time.time()
    main(mesh_size, element_size, candidates, frequency_hints, v_array, r_array, m,
         periodic=True, num_tries=40, print_frequency=True, make_figure=True,
         make_gif=False, color="#96ADFC", save_path=save_path, fig_name=fig_name,
         gif_name=gif_name,
         save_mesh=True, save_mesh_path=save_path,
         save_mesh_name="symbolic_graph.npy")
    print("Time elapsed: ", time.time() - start_time)
