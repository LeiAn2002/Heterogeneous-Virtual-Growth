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
from scipy import sparse
from joblib import Parallel, delayed


def triangle_gauss_points():
    W = np.array([0.5])
    r = np.array([1/3])
    s = np.array([1/3])
    return W, r, s


def triangle_shape_function(r, s):
    """Shape functions and derivatives with respect to the local coordinates."""
    N = np.array([(1-r-s), r, s])
    dNdr = np.array([-1.0, 1.0, 0.0])
    dNds = np.array([-1.0, 0.0, 1.0])
    return N, dNdr, dNds


def triangle_global_shape_function(dNdr, dNds, vx, vy):
    """Shape functions and derivatives with respect to the global coordinates."""
    dxdr = dNdr @ vx
    dxds = dNds @ vx
    dydr = dNdr @ vy
    dyds = dNds @ vy
    j = dxdr*dyds - dxds*dydr
    dNdx = (dNdr*dyds - dNds*dydr)/j
    dNdy = -(dNdr*dxds - dNds*dxdr)/j
    return dNdx, dNdy, j


def triangle_strain_displacement_matrix(dNdx, dNdy):
    """Shape function matrix."""
    B = np.array([
        [dNdx[0], 0, dNdx[1], 0, dNdx[2], 0],
        [0, dNdy[0], 0, dNdy[1], 0, dNdy[2]],
        [dNdy[0], dNdx[0], dNdy[1], dNdx[1], dNdy[2], dNdx[2]]
    ])
    return B


def element_stiffness_triangle(vx, vy, mat_table, D):
    """Element stiffness matrix for a triangle element."""
    ke = np.zeros((6, 6))
    W, r_coords, s_coords = triangle_gauss_points()
    thickness = mat_table["thickness"]
    for i in range(len(W)):
        r = r_coords[i]
        s = s_coords[i]
        wgt = W[i]
        
        _, dNdr, dNds = triangle_shape_function(r, s)
        dNdx, dNdy, j = triangle_global_shape_function(dNdr, dNds, vx, vy)
        B = triangle_strain_displacement_matrix(dNdx, dNdy)
        ke += B.T @ D @ B * wgt * abs(j) * thickness
    
    return ke


def quad_gauss_point(l):
    """Weights and coordinates of Gauss integration points."""
    Wgt = 1
    r_vec = np.array([-1, 1, 1, -1])*np.sqrt(3)/3
    s_vec = np.array([-1, -1, 1, 1])*np.sqrt(3)/3
    return Wgt, r_vec[l], s_vec[l]


def quad_shape_function(r, s):
    """Shape functions and derivatives with respect to the local coordinates."""
    r_vec = np.array([-1, 1, 1, -1])
    s_vec = np.array([-1, -1, 1, 1])
    N = (1+r_vec*r)*(1+s_vec*s)/4
    dNdr = r_vec*(1+s_vec*s)/4
    dNds = (1+r_vec*r)*s_vec/4
    return N, dNdr, dNds


def quad_global_shape_function(dNdr, dNds, vx, vy):
    """Shape functions and derivatives with respect to the global coordinates."""
    dxdr = dNdr @ vx
    dxds = dNds @ vx
    dydr = dNdr @ vy
    dyds = dNds @ vy
    j = dxdr*dyds - dxds*dydr
    dNdx = (dNdr*dyds - dNds*dydr)/j
    dNdy = -(dNdr*dxds - dNds*dxdr)/j
    return dNdx, dNdy, j


def quad_strain_displacement_matrix(dNdx, dNdy):
    """Shape function matrix."""
    B = np.array([
        [dNdx[0], 0, dNdx[1], 0, dNdx[2], 0, dNdx[3], 0],
        [0, dNdy[0], 0, dNdy[1], 0, dNdy[2], 0, dNdy[3]],
        [dNdy[0], dNdx[0], dNdy[1], dNdx[1], dNdy[2], dNdx[2], dNdy[3], dNdx[3]]
    ])
    return B


def elasticity_matrix(mat_table):
    """Elasticity matrix."""
    E = mat_table["E"]
    nu = mat_table["nu"]
    if mat_table["PSflag"] not in ["PlaneStress", "PlaneStrain"]:
        raise ValueError("Unsupported 'PSflag'.")
    if mat_table["PSflag"] == "PlaneStrain":
        E = E/(1-nu**2)
        nu = nu/(1-nu)
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ]) * E/(1-nu**2)
    return D


def element_stiffness_quad(vx, vy, mat_table, D):
    """Element stiffness matrix."""
    ke = np.zeros((8, 8))
    for l in np.arange(4):
        Wgt, r, s = quad_gauss_point(l)
        _, dNdr, dNds = quad_shape_function(r, s)
        dNdx, dNdy, j = quad_global_shape_function(dNdr, dNds, vx, vy)
        B = quad_strain_displacement_matrix(dNdx, dNdy)
        ke += B.T@D@B * Wgt*j*mat_table["thickness"]
    return ke


def build_tri_contrib(el_id, node_ids, nodes, mat_table, D):
    vx = nodes[node_ids, 0]
    vy = nodes[node_ids, 1]
    ke_tri = element_stiffness_triangle(vx, vy, mat_table, D)
    
    elem_dofs = []
    for n in node_ids:
        elem_dofs.append(2*n)
        elem_dofs.append(2*n+1)
    elem_dofs = np.array(elem_dofs, dtype=int)  # size=6
    
    row_sub = np.zeros(36, dtype=int)
    col_sub = np.zeros(36, dtype=int)
    data_sub = np.zeros(36, dtype=float)
    
    idx = 0
    for i in range(6):
        for j in range(6):
            row_sub[idx] = elem_dofs[i]
            col_sub[idx] = elem_dofs[j]
            data_sub[idx] = ke_tri[i,j]
            idx += 1
    
    return (row_sub, col_sub, data_sub)


def build_quad_contrib(el_id, node_ids, nodes, mat_table, D):
    vx = nodes[node_ids, 0]
    vy = nodes[node_ids, 1]
    ke_quad = element_stiffness_quad(vx, vy, mat_table, D)
    
    elem_dofs = []
    for n in node_ids:
        elem_dofs.append(2*n)
        elem_dofs.append(2*n+1)
    elem_dofs = np.array(elem_dofs, dtype=int)  # size=8
    
    row_sub = np.zeros(64, dtype=int)
    col_sub = np.zeros(64, dtype=int)
    data_sub = np.zeros(64, dtype=float)
    
    idx = 0
    for i in range(8):
        for j in range(8):
            row_sub[idx] = elem_dofs[i]
            col_sub[idx] = elem_dofs[j]
            data_sub[idx] = ke_quad[i,j]
            idx += 1
    
    return (row_sub, col_sub, data_sub)


def global_stiffness_matrix(nodes, tri_elems, quad_elems, mat_table, n_jobs=32):
    """
    Global stiffness matrix.
    Element connectivity should be counterclockwise as follows.
    3 ← 2
        ↑
    0 → 1
    """
    D = elasticity_matrix(mat_table)

    total_nodes = nodes.shape[0]
    total_dofs = 2 * total_nodes

    num_elems_tri = tri_elems.shape[0]
    num_elems_quad = quad_elems.shape[0]
    tri_results = Parallel(n_jobs=n_jobs)(
        delayed(build_tri_contrib)(el_id, tri_elems[el_id], nodes, mat_table, D)
        for el_id in range(num_elems_tri)
    )
    quad_results = Parallel(n_jobs=n_jobs)(
        delayed(build_quad_contrib)(el_id, quad_elems[el_id], nodes, mat_table, D)
        for el_id in range(num_elems_quad)
    )
    all_row = []
    all_col = []
    all_data = []
    for (r, c, d) in tri_results:
        all_row.append(r)
        all_col.append(c)
        all_data.append(d)
    for (r, c, d) in quad_results:
        all_row.append(r)
        all_col.append(c)
        all_data.append(d)

    row = np.concatenate(all_row)
    col = np.concatenate(all_col)
    data = np.concatenate(all_data)

    K_global = sparse.csc_matrix((data, (row, col)), shape=(total_dofs, total_dofs))
    return K_global
