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
import pyvista as pv
import matplotlib.pyplot as plt
from utils.linear_and_heaviside_filter import linear_filter, heaviside
from matplotlib.colors import ListedColormap
from skimage.morphology import remove_small_holes
import os
from typing import Tuple, Sequence


def plot_voxel_structure_binary(
        vol: np.ndarray,                # shape (Z,Y,X)，0=void，其余=block label
        pitch: float,
        bb_min: Tuple[float, float, float] = (0., 0., 0.),
        base_cmap: str = "Set2",        # 调色板基名
        opacity_value: float = 1.0,     # 非零体素的不透明度
        save_path: str = "",            # "" → 交互显示；否则离屏保存
        fig_name: str = "voxel_labeled.png",
        show_grid: bool = False,
        window_size: Sequence[int] = (1600, 1600)
):
    """Extract surfaces for each material label and render as meshes."""
    Nz, Ny, Nx = vol.shape
    labels = np.unique(vol)
    max_label = int(labels.max())
    if max_label == 0:
        raise ValueError("Volume only contains label 0 (void). Nothing to plot.")

    # Build a uniform ImageData grid matching the voxel layout
    grid = pv.ImageData(
        dimensions=(Nx + 1, Ny + 1, Nz + 1),
        spacing=(pitch, pitch, pitch),
        origin=bb_min,
    )
    # assign cell scalars from the volume array
    grid.cell_data["labels"] = vol.transpose(2, 1, 0).ravel(order="F").astype(np.int32)
    grid.set_active_scalars("labels")

    # Prepare a discrete colormap: white for 0, then Set2 colors for 1..max_label
    palette = plt.cm.get_cmap(base_cmap, max_label)
    colors_rgba = [(1, 1, 1, 1)]
    for i in range(max_label):
        c = palette(i)
        colors_rgba.append(c)

    cmap = ListedColormap(colors_rgba)

    # Setup plotter
    offscreen = bool(save_path)
    if offscreen:
        pv.global_theme.off_screen = True
    p = pv.Plotter(off_screen=offscreen, window_size=window_size)

    # For each material label > 0, extract its surface and add as a mesh
    for label in range(1, max_label + 1):
        # threshold to isolate this label
        sel = grid.threshold([label, label], scalars="labels")
        # convert to surface mesh
        surf = sel.extract_surface()
        # choose color from our ListedColormap
        rgba = cmap(label)
        # add the surface mesh to the plotter
        p.add_mesh(
            surf,
            color=rgba[:3],           # RGB tuple
            opacity=opacity_value,     # set opacity for the mesh
            show_edges=False,          # edges off for cleaner look
            smooth_shading=True        # enable smooth lighting
        )

    if show_grid:
        p.show_grid(color="lightgray")

    p.reset_camera()
    p.view_isometric()

    # render or save
    if offscreen:
        os.makedirs(save_path, exist_ok=True)
        p.show(auto_close=False)
        out_file = os.path.join(save_path, fig_name)
        p.screenshot(out_file)
        print(f"[INFO] Saved to {out_file}")
    else:
        p.show(title="Voxel structure (surface)")

    p.close()

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


def periodic_boundary(A):
    m, n = A.shape[0], A.shape[1]
    B = np.zeros((m + 2, n + 2, 2, 2))

    B[1:-1, 1:-1, :, :] = A

    # periodic boundary conditions
    B[0, 1:-1, :, :] = A[-1, :, :, :]
    B[-1, 1:-1, :, :] = A[0, :, :, :]
    B[1:-1, 0, :, :] = A[:, -1, :, :]
    B[1:-1, -1, :, :] = A[:, 0, :, :]

    return B


def plot_microstructure_2d(m, full_mesh, all_elems, block_library,
                           v_array, r_array, periodic, color="#96ADFC",
                           save_path="", fig_name="microstructure.jpg"):
    rows, cols = full_mesh.shape

    k = 0
    thickness_matrices = np.zeros((rows, cols, 2, 2))

    # get the initial thickness matrices
    for y in range(full_mesh.shape[0]):
        for x in range(full_mesh.shape[1]):
            block = full_mesh[y][x]
            parent = block[:block.index(" ")]
            if parent == "void":
                k += 1
                continue
            suffix_str = block[block.index(" ") + 1:]
            rotation = int(suffix_str)
            elem_id = all_elems[k]
            v_range = v_array[elem_id]
            random_radius = r_array[elem_id]
            block_class = block_library.create_block(parent, m, v_range, rotation, random_radius)
            thickness_matrices[y, x] = block_class.get_thickness()
            k += 1

    if periodic:
        # get the connectivity-ganranteeing thickness matrices under periodic boundary conditions
        periodic_thickness_matrices = periodic_boundary(thickness_matrices)
        rows_periodic_matrix, cols_periodic_matrix = periodic_thickness_matrices.shape[0], periodic_thickness_matrices.shape[1]
        for y in range(rows_periodic_matrix):
            for x in range(cols_periodic_matrix):
                if x < cols_periodic_matrix - 1:
                    if periodic_thickness_matrices[y, x, 1, 1] != 0 and periodic_thickness_matrices[y, x + 1, 1, 0] != 0:
                        avg = (periodic_thickness_matrices[y, x, 1, 1] + periodic_thickness_matrices[y, x + 1, 1, 0]) / 2
                        periodic_thickness_matrices[y, x, 1, 1] = avg
                        periodic_thickness_matrices[y, x + 1, 1, 0] = avg
                if y < rows_periodic_matrix - 1:
                    if periodic_thickness_matrices[y, x, 0, 0] != 0 and periodic_thickness_matrices[y + 1, x, 0, 1] != 0:
                        avg = (periodic_thickness_matrices[y, x, 0, 0] + periodic_thickness_matrices[y + 1, x, 0, 1]) / 2
                        periodic_thickness_matrices[y, x, 0, 0] = avg
                        periodic_thickness_matrices[y + 1, x, 0, 1] = avg
        thickness_matrices = periodic_thickness_matrices[1:-1, 1:-1]

    else:
        # get the connectivity-ganranteeing thickness matrices
        for y in range(thickness_matrices.shape[0]):
            for x in range(thickness_matrices.shape[1]):
                if x < full_mesh.shape[1] - 1:
                    if periodic_thickness_matrices[y, x, 1, 1] != 0 and periodic_thickness_matrices[y, x + 1, 1, 0] != 0:
                        avg = (thickness_matrices[y, x, 1, 1] + thickness_matrices[y, x + 1, 1, 0]) / 2
                        thickness_matrices[y, x, 1, 1] = avg
                        thickness_matrices[y, x + 1, 1, 0] = avg
                if y < full_mesh.shape[0] - 1:
                    if periodic_thickness_matrices[y, x, 0, 0] != 0 and periodic_thickness_matrices[y + 1, x, 0, 1] != 0:
                        avg = (thickness_matrices[y, x, 0, 0] + thickness_matrices[y + 1, x, 0, 1]) / 2
                        thickness_matrices[y, x, 0, 0] = avg
                        thickness_matrices[y + 1, x, 0, 1] = avg

    block_size = 55
    final_height = rows * block_size
    final_width = cols * block_size
    final_raster = np.zeros((final_height, final_width), dtype=np.uint8)
    color_label_matrix = np.ones((rows, cols))
    block_name_to_label = {"star": 1, "gripper": 2, "arrow": 3, "V": 4, "O": 5}
    # block_name_to_label = {"star": 2}
    # block_name_to_label = {"H": 1, "T": 2, "V": 3, "TT": 4}  # Initialize with void label
    # next_label = 1

    k = 0
    for y in range(full_mesh.shape[0]):
        for x in range(full_mesh.shape[1]):
            block = full_mesh[y][x]
            parent = block[:block.index(" ")]
            if parent == "void":
                k += 1
                continue
            suffix_str = block[block.index(" ") + 1:]
            rotation = int(suffix_str)
            elem_id = all_elems[k]
            v_range = v_array[elem_id]
            random_radius = r_array[elem_id]

            # if parent not in block_name_to_label:
            #     block_name_to_label[parent] = next_label
            #     next_label += 1
            label_id = block_name_to_label[parent]

            block_class = block_library.create_block(parent, m, v_range, rotation, random_radius)
            block_matrix = block_class.generate_block_shape(thickness_matrices[y, x])
            top = y * block_size
            left = x * block_size
            final_raster[top:top+block_size, left:left+block_size] = block_matrix

            color_label_matrix[y, x] = label_id
            k += 1

    final_raster = linear_filter(final_raster, 4)
    final_raster = heaviside(final_raster, 512)
    # final_raster = final_raster.astype(bool)
    # final_raster = remove_small_holes(final_raster, area_threshold=100)
    # final_raster = final_raster.astype(float)

    colored_final_raster = final_raster.copy()

    for y in range(full_mesh.shape[0]):
        for x in range(full_mesh.shape[1]):
            top = y * block_size
            left = x * block_size
            label_id = color_label_matrix[y, x]
            colored_final_raster[top:top+block_size, left:left+block_size] *= label_id

    num_labels = 5  # 1..(next_label-1) are real block labels
    color_list = []
    color_list.append([1, 1, 1])  # background => white
    palette = plt.cm.get_cmap("Set2", num_labels)
    for i in range(num_labels):
        color_list.append(palette(i))

    cmap = ListedColormap(color_list)

    plt.figure(figsize=(32, 32))
    plt.axis("off")
    plt.imshow(colored_final_raster, cmap=cmap, origin="upper")
    if save_path:
        plt.savefig(save_path + fig_name, bbox_inches="tight")
    plt.close()

    return final_raster


ZMIN, ZMAX, YMIN, YMAX, XMIN, XMAX = range(6)


def plot_microstructure_3d(
    m,
    full_mesh,                  # (Nz, Ny, Nx) of "parent suffix"
    uid2oid,
    all_elems,                  # len = Nz*Ny*Nx
    block_library,
    v_array,                    # (Ne,2)
    r_array,                    # (Ne,)
    periodic=True,
    save_path="",
    fig_name="microstructure_3d.png",
):
    """
    Assemble all voxel blocks into one global volume, colour by parent name,
    and render with PyVista.

    Returns
    -------
    volume : np.ndarray(uint8)  shape (Ztot, Ytot, Xtot)  – 0=void, 1..n=labels
    """
    # print(full_mesh)
    # print(uid2oid)
    Nz, Ny, Nx = full_mesh.shape

    parent_set = sorted({cell.split(" ")[0] for cell in full_mesh.ravel()
                         if not cell.startswith("void")})
    parent2lbl = {p: i + 1 for i, p in enumerate(parent_set)}
    # n_lbl = len(parent_set)
    # color_list = [[1, 1, 1]] + [plt.get_cmap("Set2", n_lbl)(i)[:3]
    #                              for i in range(n_lbl)]
    # cmap = ListedColormap(color_list)

    thickness_arr = np.zeros((Nz, Ny, Nx, 6), dtype=np.float32)
    block_meta = []
    k = 0
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                block = full_mesh[z][y][x]
                parent = block[:block.index(" ")]
                if parent == "void":
                    k += 1
                    continue
                suffix_str = block[block.index(" ") + 1:]
                rotation = int(suffix_str)
                rotation_oid = uid2oid[parent][rotation]
                # print(rotation, rotation_oid)
                eid = all_elems[k]
                v_rng = v_array[eid]
                rand_r = r_array[eid]
                blk = block_library.create_block(parent, m, v_rng,
                                                 rotation_oid, rand_r)
                thickness_arr[z, y, x] = blk.get_thickness()
                block_meta.append((z, y, x, parent, rotation_oid, v_rng, rand_r))
                k += 1

    def _avg(a, b):
        mask = (a > 0) & (b > 0)
        avg = 0.5 * (a[mask] + b[mask])
        a[mask] = b[mask] = avg

    _avg(thickness_arr[:, :, :-1, XMAX], thickness_arr[:, :, 1:, XMIN])
    _avg(thickness_arr[:, :-1, :, YMAX], thickness_arr[:, 1:, :, YMIN])
    _avg(thickness_arr[:-1, :, :, ZMAX], thickness_arr[1:, :, :, ZMIN])

    if periodic:
        _avg(thickness_arr[:, :, 0, XMIN], thickness_arr[:, :, -1, XMAX])
        _avg(thickness_arr[:, 0, :, YMIN], thickness_arr[:, -1, :, YMAX])
        _avg(thickness_arr[0, :, :, ZMIN], thickness_arr[-1, :, :, ZMAX])

    # np.savetxt(os.path.join(save_path, "thickness_arr.txt"),
    #            thickness_arr.reshape(-1, 6), fmt="%.4f")

    sample_z, sample_y, sample_x, *_ = block_meta[0]
    sample_blk = block_library.create_block(
        block_meta[0][3], m, block_meta[0][5], block_meta[0][4],
        block_meta[0][6])
    bz, by, bx = sample_blk.generate_block_shape(
        thickness_arr[sample_z, sample_y, sample_x]).shape
    assert bz == by == bx
    bsize = bz

    vol = np.zeros((Nz*bsize, Ny*bsize, Nx*bsize), dtype=np.uint8)

    for z, y, x, parent, rot, v_rng, rand_r in block_meta:
        blk = block_library.create_block(parent, m, v_rng, rot, rand_r)
        vox = blk.generate_block_shape(
            thickness_arr[z, y, x])
        lbl = parent2lbl[parent]
        vox *= lbl
        z0, y0, x0 = z*bsize, y*bsize, x*bsize
        vol[z0:z0+bsize, y0:y0+by, x0:x0+bx] = np.maximum(
            vol[z0:z0+bsize, y0:y0+by, x0:x0+bx], vox)
        
    # save vol as txt
    # if save_path:
    #     os.makedirs(save_path, exist_ok=True)
    #     np.savetxt(os.path.join(save_path, "voxels.txt"), vol.reshape(-1), fmt="%d")
    # print(np.unique(vol))
    # np.savetxt("vol.txt", vol.flatten(), fmt='%.6f')

    plot_voxel_structure_binary(
        vol,               # 0/1 ndarray
        pitch=0.04,
        bb_min=(-1, -1, -1),
        opacity_value=1.0,
        save_path=save_path,
        fig_name=fig_name,
    )

    return vol


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
