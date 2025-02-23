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

import os
import numpy as np
import sys

from virtual_growth.plot_mesh import plot_mesh
from utils.array_list_operations import find_indices
from utils.remove_repeated_nodes import remove_repeated_nodes


def generate_fem_mesh_2d(symbolic_mesh, block_size, data_path,
                         all_elems: np.ndarray, v_array: np.ndarray, rescale=False, geometry=(1, 1),
                         check_with_pyvista=False, save_mesh=True, mesh_path=""):
    """Generate the fem mesh of the input symbolic graph."""
    # Load the data
    # all_fem_elements = np.load(data_path + "/0.3/unique_block_fem_elements.npy",
    #                            allow_pickle=True)
    # all_fem_nodes = np.load(data_path + "unique_block_fem_nodes.npy",
    #                         allow_pickle=True)
    names = np.load(data_path + "unique_names.npy")

    # Extract the mesh information
    num_elem_y, num_elem_x = symbolic_mesh.shape
    total_cells = num_elem_y * num_elem_x
    # indices = find_indices(names, symbolic_mesh.flatten())
    global_nodes_list = []
    global_elements_list = []

    # elements = all_fem_elements[indices]
    # print(elements)
    # num_elems = np.array(list(map(len, elements)))
    # elements = np.vstack(elements)[:, 1:]
    # # print(elements)
    # # print(indices)
    # sys.exit()

    # nodes = all_fem_nodes[indices]
    # num_nodes = np.array(list(map(len, nodes)))
    # nodes = np.vstack(nodes)

    v_cache = {}

    def load_v_data(v_val):
        subdir = os.path.join(data_path, str(v_val))
        if not os.path.exists(subdir):
            raise FileNotFoundError(f"No subdir found for v={v_val} at {subdir}")
        # names_v = np.load(os.path.join(subdir, "unique_names.npy"), allow_pickle=True)
        fem_elems_v = np.load(os.path.join(subdir, "unique_block_fem_elements.npy"), allow_pickle=True)
        fem_nodes_v = np.load(os.path.join(subdir, "unique_block_fem_nodes.npy"), allow_pickle=True)
        return fem_elems_v, fem_nodes_v
    
    flat_mesh = symbolic_mesh.flatten()

    for i_cell in range(total_cells):
        block_str = flat_mesh[i_cell]
        elem_id = all_elems[i_cell]
        v_val = v_array[elem_id]

        if v_val not in v_cache:
            fem_elems_v, fem_nodes_v = load_v_data(v_val)
            v_cache[v_val] = {
                "elems": fem_elems_v,
                "nodes": fem_nodes_v
            }
        
        data_v = v_cache[v_val]
        # names_v = data_v["names"]
        fem_elems_v = data_v["elems"]
        fem_nodes_v = data_v["nodes"]

        idx_in_v = names.tolist().index(block_str)
        local_elems = fem_elems_v[idx_in_v]
        local_nodes = fem_nodes_v[idx_in_v].copy()
        # y_idx, x_idx = divmod(i_cell, num_elem_x)
        # local_nodes[:, 0] += x_idx * block_size
        # local_nodes[:, 1] -= y_idx * block_size

        # local_elems[:, 1:] += node_offset

        global_nodes_list.append(local_nodes)
        global_elements_list.append(local_elems)

        # node_offset += local_nodes.shape[0]
    num_elems = np.array(list(map(len, global_elements_list)))
    num_nodes = np.array(list(map(len, global_nodes_list)))
    elements = np.vstack(global_elements_list)[:, 1:]
    nodes = np.vstack(global_nodes_list)

    # Modify nodes
    offset_x = np.tile(np.arange(num_elem_x), num_elem_y)
    offset_y = np.tile(np.arange(num_elem_y).reshape(-1, 1), (1, num_elem_x)).flatten()
    offset_x = np.repeat(offset_x, num_nodes) * block_size
    offset_y = np.repeat(offset_y, num_nodes) * block_size
    nodes[:, 0] += offset_x
    nodes[:, 1] -= offset_y

    # Shift nodes to the first quadrant
    nodes[:, 0] += block_size/2
    nodes[:, 1] += block_size * (num_elem_y-1/2)

    # all_nodes = np.vstack(global_nodes_list)
    # all_elements = np.vstack(global_elements_list)
    # all_nodes[:, 0] += block_size/2.0
    # all_nodes[:, 1] += block_size * (num_elem_y - 0.5)

    if rescale:
        nodes[:, 0] = nodes[:, 0] / max(nodes[:, 0]) * geometry[0]
        nodes[:, 1] = nodes[:, 1] / max(nodes[:, 1]) * geometry[1]

    # Modify elements
    offset = np.hstack((0, np.cumsum(num_nodes)[:-1]))
    offset = np.repeat(offset, num_elems).reshape(-1, 1)
    elements += offset

    nodes, elements = remove_repeated_nodes(nodes, elements, precision=12)
    elements = np.hstack((np.ones((elements.shape[0], 1), dtype=int)*4, elements))
    cell_types = np.ones(elements.shape[0], dtype=int) * 9
    num_elements, num_nodes = elements.shape[0], nodes.shape[0]
    print(f"Number of elements: {num_elements}, Number of nodes: {num_nodes}\n")

    if check_with_pyvista:
        plot_mesh(elements, cell_types, nodes, plot_box=None, view_xy=True,
                  show_edges=True, line_width=1, fig_path=mesh_path, figsize=800,
                  fig_name="fem_mesh.jpg")

    if save_mesh:
        if mesh_path != "" and not os.path.exists(mesh_path):
            os.makedirs(mesh_path)
        np.savez(mesh_path+"fem_mesh.npz", nodes=nodes, elements=elements[:, 1:])

    return elements, cell_types, nodes