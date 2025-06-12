"""
Batch convert: read each <name>.xdmf (+<name>.h5), L2-project to DG0
and save as <name>_DG0.xdmf. Works without XDMFFile.read_function.
"""

import h5py
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from pathlib import Path
from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_vector, assemble_matrix


# -------------- helper 1: fallback loader (see section 2) ---------- #
def load_field_from_h5(mesh, h5_path, func_name=None):
    with h5py.File(h5_path, "r") as h5:
        fn_group = list(h5["Function"].keys())[0] if func_name is None else func_name
        data = h5[f"Function/{fn_group}/0"][()]

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    is_dg0    = (data.size == num_cells)

    if is_dg0:                                   # reorder by (y,x)
        if mesh.topology.connectivity(mesh.topology.dim, 0) is None:
            mesh.topology.create_connectivity(mesh.topology.dim, 0)
        conn  = mesh.topology.connectivity(mesh.topology.dim, 0)
        cells = np.array([conn.links(i)
                          for i in range(mesh.topology.index_map(mesh.topology.dim).size_local)])
        cent  = mesh.geometry.x[cells].mean(axis=1)
        sort_idx = np.lexsort((cent[:, 0], cent[:, 1]))
        data = data[sort_idx]

    V = fem.functionspace(mesh, ("DG", 0) if is_dg0 else ("CG", 1))
    f = fem.Function(V, name=fn_group)
    loc0, loc1 = V.dofmap.index_map.local_range
    f.vector.array[:] = data[loc0:loc1].flatten()
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                         mode=PETSc.ScatterMode.FORWARD)
    return f, is_dg0, num_cells


# -------------- helper 2: L2 projector ----------------------------- #
def l2_project(src, V_dst):
    u, v = ufl.TrialFunction(V_dst), ufl.TestFunction(V_dst)
    A = assemble_matrix(fem.form(ufl.inner(u, v) * ufl.dx)); A.assemble()
    b = assemble_vector(fem.form(ufl.inner(src, v) * ufl.dx))
    dst = fem.Function(V_dst, name=f"{src.name}_DG0")
    ksp = PETSc.KSP().create(V_dst.mesh.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    ksp.solve(b, dst.vector)
    dst.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                           mode=PETSc.ScatterMode.FORWARD)
    return dst

# -------------- main loop ------------------------------------------ #
input_files = ["./datas/data_cloak_new/rho_field.xdmf",
               "./datas/data_cloak_new/ksi_field_1.xdmf",
               "./datas/data_cloak_new/ksi_field_2.xdmf",
               "./datas/data_cloak_new/ksi_field_3.xdmf",
               "./datas/data_cloak_new/ksi_field_4.xdmf",
               "./datas/data_cloak_new/ksi_field_5.xdmf",
               "./datas/data_cloak_new/vf_field.xdmf"]  # list your files here
suffix_out    = "_DG0.xdmf"

if __name__ == "__main__":
    for xfile in input_files:
        stem = Path(xfile).stem
        h5file = Path(xfile).with_suffix(".h5")
        if not h5file.exists():
            raise FileNotFoundError(f"{h5file} missing.")

        # (1) read mesh
        with io.XDMFFile(MPI.COMM_WORLD, xfile, "r") as xdmf:
            mesh = xdmf.read_mesh()

        # (2) read field via h5py
        f_src, already_dg0, _ = load_field_from_h5(mesh, h5file)

        # (3) if already DG0 -> keep; else L2-project
        V_dg0 = f_src.function_space if already_dg0 else fem.functionspace(mesh, ("DG", 0))
        f_dg0 = f_src if already_dg0 else l2_project(f_src, V_dg0)

        # (4) write result
        out = Path(xfile).with_stem(stem + suffix_out.replace(".xdmf", "")).with_suffix(".xdmf")
        with io.XDMFFile(MPI.COMM_WORLD, out, "w") as xout:
            xout.write_mesh(mesh)
            xout.write_function(f_dg0, t=0.0)

        if MPI.COMM_WORLD.rank == 0:
            print(f"[OK] {Path(xfile).name}  ->  {out.name}")
