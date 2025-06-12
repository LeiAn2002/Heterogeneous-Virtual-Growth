import numpy as np
import os

from virtual_growth.main import main
from blocks.block_mesh_2d import generate_mesh, load_msh_with_meshio
from homogenization.homogenization_2d import homogenized_elasticity_matrix_2d

output_file = "formal_test_data.npz"

num_of_groups = 1000
runs_per_group = 1

save_path   = "designs/2d/"
design_path = os.path.join(save_path, "symbolic_graph.npy")
geo_file    = os.path.join(save_path, "mesh.geo")
msh_file    = os.path.join(save_path, "mesh.msh")

v_bias = 0.05
mesh_number = 1
element_number = 10
mesh_size = (mesh_number, mesh_number)
element_size = (element_number, element_number)
m = 0.75
candidates = ["star", "gripper", "T", "V", "O"]
num_elems = np.prod(mesh_size)

np.random.seed(12345)
data = np.zeros((num_of_groups, num_elems, len(candidates)))
v_data = np.random.uniform(0.35, 0.7, size=(num_of_groups,))

for i in range(num_of_groups):
    random_values = np.random.uniform(0, 1, (num_elems, len(candidates)))
    normalized_values = random_values / np.sum(random_values)
    data[i] = normalized_values

if os.path.exists(output_file):
    existing_data = np.load(output_file)
    all_inputs = existing_data["inputs"].tolist()
    all_labels = existing_data["labels"].tolist()
else:
    all_inputs = []
    all_labels = []

for i in range(num_of_groups):
    frequency_hints = data[i]
    v = v_data[i]
    selected_elements_all = []
    for j in range(runs_per_group):
        try:
            v_array = np.random.uniform(low=v-v_bias, high=v+v_bias, size=(num_elems, 2))
            r_array = np.random.uniform(low=0.05, high=0.05, size=(num_elems,))
            main(
                    mesh_size, element_size, 
                    candidates,
                    frequency_hints,
                    v_array,
                    r_array,
                    m,
                    periodic=True,
                    num_tries=40,
                    print_frequency=False,
                    make_figure=True,
                    make_gif=False,
                    save_path=save_path,
                    fig_name="symbolic_graph.jpg",
                    gif_name="symbolic_graph.gif",
                    save_mesh=True,
                    save_mesh_path=save_path,
                    save_mesh_name="symbolic_graph.npy"
                )

            generate_mesh(design_path, geo_file, msh_file)

            nodes, tri_elems, quad_elems = load_msh_with_meshio(msh_file)
            mat_table = {
                "E": 2.41,
                "nu": 0.35,
                "PSflag": "PlaneStress",
                "RegMesh": False,
                "thickness": 1.0,
            }
            K_eps = homogenized_elasticity_matrix_2d(
                nodes, tri_elems, quad_elems, mat_table
            )

            upper_triangle_indices = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            selected_elements = [K_eps[i, j] for i, j in upper_triangle_indices]
            selected_elements_all.append(selected_elements)

        except Exception as e:
            print(f"Error in group {i + 1}, run {j + 1}: {e}")
            continue

    if not selected_elements_all:
        print(f"Group {i+1}: all runs failed ‑ skipping.")
        continue

    selected_elements_avg = np.mean(selected_elements_all, axis=0)
    augmented_inputs = np.hstack([
        frequency_hints,
        np.full((num_elems, 1), v)
    ])
    all_inputs.append(augmented_inputs.tolist())
    all_labels.append(selected_elements_avg.tolist())

    np.savez(output_file, inputs=np.array(all_inputs), labels=np.array(all_labels))
    print(f"Data saved for group {i + 1}/1000")

print("All data processing completed and saved.")
