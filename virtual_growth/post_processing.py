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

from collections import Counter
import numpy as np
import pyvista

from utils.remove_repeated_nodes import remove_repeated_nodes


def compute_final_frequency(block_count, num_elem, aug_candidates, candidates):
    """This function is used to compute frequency distribution of designs."""
    reduced_list = aug_candidates.copy()
    for n, item in enumerate(reduced_list):
        reduced_list[n] = item[:item.index(" ")]
    counter = dict(Counter(reduced_list))

    frequency = np.zeros((num_elem, len(candidates)))
    k = 0
    for n, item in enumerate(candidates):
        frequency[:, n] = np.sum(block_count[:, k:k+counter[item]], axis=1)
        k += counter[item]
    frequency /= np.sum(frequency, axis=1).reshape(-1, 1)
    return frequency


def plot_microstructure_2d(m, full_mesh, all_elems, block_library,
                           v_array, solid=[], void=[], color="#96ADFC",
                           save_path="", fig_name="microstructure.jpg"):
    block_size = 2 * m
    rows, cols = full_mesh.shape
    cell_nodes = [[None for _ in range(cols)] for _ in range(rows)]
    cell_elems = [[None for _ in range(cols)] for _ in range(rows)]

    node_count = 0
    element_count = np.zeros(full_mesh.size, dtype=int)
    element_list, node_list = [], []
    k = 0
    thickness_matrices = np.zeros((full_mesh.shape[0], full_mesh.shape[1], 2, 2))

    # get the initial thickness matrices
    for y in range(full_mesh.shape[0]):
        for x in range(full_mesh.shape[1]):
            block = full_mesh[y][x]
            parent = block[:block.index(" ")]
            suffix_str = block[block.index(" ") + 1:]
            rotation = int(suffix_str)
            elem_id = all_elems[k]
            v_val = v_array[elem_id]
            block_class = block_library.create_block(parent, m, v_val, rotation)
            thickness_matrices[y, x] = block_class.get_thickness()
            k += 1

    # get the connectivity-ganranteeing thickness matrices
    for y in range(full_mesh.shape[0]):
        for x in range(full_mesh.shape[1]):
            if x < full_mesh.shape[1] - 1:
                avg = (thickness_matrices[y, x, 1, 0] + thickness_matrices[y, x + 1, 0, 0]) / 2
                thickness_matrices[y, x, 1, 0] = avg
                thickness_matrices[y, x + 1, 0, 0] = avg
            if y < full_mesh.shape[0] - 1:
                avg = (thickness_matrices[y, x, 0, 1] + thickness_matrices[y + 1, x, 1, 1]) / 2
                thickness_matrices[y, x, 0, 1] = avg
                thickness_matrices[y + 1, x, 1, 1] = avg

    k = 0
    for y in range(full_mesh.shape[0]):
        for x in range(full_mesh.shape[1]):
            block = full_mesh[y][x]
            parent = block[:block.index(" ")]
            suffix_str = block[block.index(" ") + 1:]
            rotation = int(suffix_str)
            elem_id = all_elems[k]
            v_val = v_array[elem_id]
            block_class = block_library.create_block(parent, m, v_val, rotation)
            elements = block_class.generate_elements().copy()
            nodes = block_class.generate_nodes(thickness_matrices[y, x]).copy()
            nodes[:, 0] += block_size * x
            nodes[:, 1] -= block_size * y  # Note here it should minus
            cell_nodes[y][x] = nodes
            cell_elems[y][x] = elements
            k += 1

    k = 0
    for y in range(rows):
        for x in range(cols):
            local_nodes = cell_nodes[y][x]
            local_elems = cell_elems[y][x].copy()
            local_elems[:, 1:] += node_count
            node_list.extend(local_nodes.tolist())
            element_list.extend(local_elems.tolist())

            element_count[k] = local_elems.shape[0]
            node_count += local_nodes.shape[0]
            k += 1

    elements = np.array(element_list)
    nodes = np.array(node_list)
    nodes, elements = remove_repeated_nodes(nodes, elements[:, 1:], precision=6)
    elements = np.hstack((
        np.full((elements.shape[0], 1), elements.shape[1]), elements,
    )).astype(int)
    cell_types = np.full(elements.shape[0], 9, dtype=int)

    pyvista.OFF_SCREEN = True
    pyvista.set_plot_theme("document")
    pyvista.start_xvfb()
    figsize = 2000
    plotter = pyvista.Plotter(window_size=[figsize, figsize])
    grid = pyvista.UnstructuredGrid(elements, cell_types, nodes)
    plotter.add_mesh(grid, color=color, lighting=True,
                     show_edges=False, show_scalar_bar=False)

    plotter.background_color = "white"
    plotter.view_xy()
    plotter.show_axes()

    plotter.screenshot(save_path+fig_name, window_size=[figsize, figsize])
    plotter.close()

    return elements, cell_types, nodes, element_count


def plot_microstructure_3d(full_mesh, block_lib, block_nodes, names,
                           block_size, solid=[], void=[], color="#96ADFC",
                           save_path="", fig_name="microstructure.jpg"):
    node_count = 0
    k = 0
    element_count = np.zeros(full_mesh.size, dtype=int)
    element_list, node_list = [], []

    for z in range(full_mesh.shape[0]):
        for y in range(full_mesh.shape[1]):
            for x in range(full_mesh.shape[2]):
                block = full_mesh[z][y][x]
                parent = block[:block.index(" ")]
                index = names.tolist().index(block)

                elements = block_lib[parent]["elements"].copy()
                elements[:, 1:] += node_count
                nodes = block_nodes[index].copy()

                nodes[:, 0] -= block_size * x  # Note here it should minus
                nodes[:, 1] += block_size * y
                nodes[:, 2] += block_size * z
                node_count += nodes.shape[0]

                element_list.extend(elements.tolist())
                node_list.extend(nodes.tolist())

                element_count[k] = elements.shape[0]
                k += 1

    elements = np.array(element_list)
    nodes = np.array(node_list)
    nodes, elements = remove_repeated_nodes(nodes, elements[:, 1:], precision=6)
    elements = np.hstack((
        np.full((elements.shape[0], 1), elements.shape[1]), elements,
    )).astype(int)
    cell_types = np.full(elements.shape[0], 12, dtype=int)

    pyvista.OFF_SCREEN = True
    pyvista.set_plot_theme("document")
    pyvista.start_xvfb()
    figsize = 2000
    plotter = pyvista.Plotter(window_size=[figsize, figsize])
    grid = pyvista.UnstructuredGrid(elements, cell_types, nodes)
    plotter.add_mesh(grid, color=color, lighting=True,
                     show_edges=False, show_scalar_bar=False)

    plotter.background_color = "white"
    plotter.show_axes()

    plotter.screenshot(save_path+fig_name, window_size=[figsize, figsize])
    plotter.close()

    return elements, cell_types, nodes, element_count


def plot_microstructure_gif(fill_sequence, elements, cell_types, nodes,
                            element_count, dim=3, color="#96ADFC",
                            save_path="", gif_name="microstructure.gif"):
    pyvista.OFF_SCREEN = True
    pyvista.set_plot_theme("document")
    pyvista.start_xvfb()
    figsize = 2000
    plotter = pyvista.Plotter(window_size=[figsize, figsize])

    fill_sequence = fill_sequence.astype(int)
    start = np.sum(element_count[:fill_sequence[0]]).astype(int)
    end = np.sum(element_count[:fill_sequence[0]+1]).astype(int)
    elem_list = np.arange(start, end).tolist()
    grid = pyvista.UnstructuredGrid(
        elements[elem_list], cell_types[elem_list], nodes)
    plotter.add_mesh(grid, color=color, lighting=True, show_edges=False,
                     show_scalar_bar=False, name="mesh_actor")
    plotter.background_color = "white"
    if dim == 2:
        plotter.view_xy()
    plotter.open_gif(save_path+gif_name, framerate=24)
    plotter.write_frame()

    for n in fill_sequence[1:]:
        start = np.sum(element_count[:n]).astype(int)
        end = np.sum(element_count[:n+1]).astype(int)
        elem_list.extend(np.arange(start, end).tolist())
        grid = pyvista.UnstructuredGrid(
            elements[elem_list], cell_types[elem_list], nodes)
        plotter.add_mesh(grid, color=color, lighting=True,
                         show_edges=False, show_scalar_bar=False,
                         name="mesh_actor")
        plotter.write_frame()
    plotter.close()
