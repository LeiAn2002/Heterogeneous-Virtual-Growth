import numpy as np
from utils.remove_repeated_nodes import remove_repeated_nodes


def process_raw_mesh(nodes_list, elements_list):
    """
    Remove repeated nodes and convert the mesh info to the Pyvista format.
    """
    all_nodes = np.vstack(nodes_list)
    all_elements = np.zeros(np.vstack(elements_list).shape)

    elem_count = 0
    node_count = 0
    for nodes, elements in zip(nodes_list, elements_list):
        elements += node_count
        all_elements[elem_count:elem_count+elements.shape[0]] = elements
        elem_count += elements.shape[0]
        node_count += nodes.shape[0]

    nodes, elements = remove_repeated_nodes(
        all_nodes, all_elements.astype(int), precision=12)
    nodes = np.hstack((nodes, np.zeros((nodes.shape[0], 1))))
    elements = np.hstack((np.ones((elements.shape[0], 1))*elements.shape[1],
                         elements)).astype(int)
    cell_types = np.ones(elements.shape[0], dtype=int) * 9
    return elements, cell_types, nodes


def coord_transformation(r, s, vx, vy):
    """
    Compute coordinates in a rectangular field after tranformation.
    Inputs:
        r, s: Coordinates in the natural coordinate system.
        vx, vy: Coordinates of four vertexes of the transformed field.
    Outputs:
        coord_x, coord_y: Coordinates in the transformed field.
    """
    r_vec = np.array([-1, 1, 1, -1])
    s_vec = np.array([-1, -1, 1, 1])
    temp1 = r.reshape(-1, 1) * r_vec
    temp2 = s.reshape(-1, 1) * s_vec
    N = (1 + temp1) * (1 + temp2) / 4
    coord_x = np.sum(N * vx, axis=1)
    coord_y = np.sum(N * vy, axis=1)
    return coord_x, coord_y
