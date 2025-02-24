import time
import datetime
import pytz
from tqdm import tqdm
import numpy as np
import os

from virtual_growth.adjacency_rules import (
    augment_candidates, find_admissible_blocks_2d, find_admissible_blocks_3d
)
from virtual_growth.post_processing import (
    compute_final_frequency,
    plot_microstructure_2d,
    plot_microstructure_3d,
    plot_microstructure_gif,
)
from utils.array_list_operations import find_indices
from virtual_growth.pair_rules_2d import PairRules2D


class VirtualGrowthEngine:
    """
    An OOP-based engine for 2D or 3D virtual growth.

    - Accepts a BlockLibrary, from which we create blocks or retrieve info.
    - Can generate pair adjacency rules.
    - Runs the growth algorithm and does post-processing, all in one place.
    """

    def __init__(self, block_library):
        """
        :param block_library: An instance of your existing BlockLibrary class.
        """
        self.block_library = block_library

        # Will store data produced by generate_pair_rules
        self.all_unique_blocks = None
        self.all_extended_block_names = None
        self.encoded_rotation_table = None
        self.encoded_rules = None
        self.encoded_special_rules = None

        # Will store final adjacency arrays or other decoded info needed for growth
        self.names = None
        self.rules = None
        self.rotation_table = None
        self.special_rules = None

        # Additional placeholders
        self.dim = None
        self.num_elems = None
        self.num_cells_elem = None
        self.num_cells = None
        self.x_cell = None
        self.y_cell = None
        self.z_cell = None
        self.all_elems = None

    def generate_pair_rules(self, block_names):
        """
        Generate adjacency rules, rotation tables, etc., using PairRules2D.
        Stores the results as class attributes for subsequent use in run_growth.
        """
        pair_rule_gen = PairRules2D(self.block_library)

        (
            self.all_unique_blocks,
            self.all_extended_block_names,
            self.encoded_rotation_table,
            self.encoded_rules,
            self.encoded_special_rules
        ) = pair_rule_gen.generate_rules(block_names)

        # Optionally decode or reorganize these if needed.
        # For example, if in your actual adjacency logic you rely on "names" being a numpy array:
        self.names = np.array(self.all_extended_block_names, dtype=object)
        self.rules = self.encoded_rules
        self.rotation_table = self.encoded_rotation_table
        self.special_rules = self.encoded_special_rules

    def run_growth(
        self,
        mesh_size,
        elem_size,
        candidates,
        frequency_hints,
        v_array,
        m,
        periodic=True,
        num_tries=1,
        print_frequency=True,
        make_figure=True,
        make_gif=True,
        color="#96ADFC",
        save_path="",
        fig_name="microstructure.jpg",
        gif_name="microstructure.mp4",
        save_mesh=False, save_mesh_path="",
        save_mesh_name="symbolic_graph.npy"
    ):
        """
        The main function for running the virtual growth process.
        """
        start_time_info = "Program start time: " + datetime.datetime.now(
            pytz.timezone("America/Chicago")
        ).strftime("%Y-%m-%d %H:%M:%S")
        print(start_time_info)
        start_time = time.time()

        # Basic checks
        if frequency_hints.shape[1] != len(candidates):
            raise ValueError("Candidate blocks and frequency hints are incompatible.")

        row_sums = np.sum(frequency_hints, axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Sum of frequency hints in each row must be 1.")

        if len(mesh_size) != len(elem_size):
            raise ValueError("Dimensions of mesh and element are incompatible.")

        # Compute some parameters
        self._prepare_dimensions(mesh_size, elem_size)

        names = self.names
        rules = self.rules
        rotation_table = self.rotation_table
        special_rules = self.special_rules

        if names is None or rules is None or rotation_table is None or special_rules is None:
            raise RuntimeError(
                "Pair rules have not been generated. Please call generate_pair_rules(...) first."
            )
        
        # Generate augmented candidates
        aug_candidates, aug_frequency_hints = augment_candidates(
            candidates, frequency_hints, names
        )
        aug_candidates_encoded = find_indices(names, aug_candidates)
        aug_candidates_encoded_ref = np.hstack((aug_candidates_encoded, -1))

        # Add "unfilled" and "wall" cases to rules and rotation_table
        rules, aug_candidates_encoded, rotation_table = self._add_unfilled_and_wall(rules, aug_candidates_encoded, rotation_table)

        # Attempt growth multiple times
        admissible = False
        for n_try in range(num_tries):
            try:
                # Prepare some arrays
                # For 2D:
                if self.dim == 2:
                    full_mesh = np.full((self.num_cells_y, self.num_cells_x), -1, dtype=object)
                    aug_full_mesh = np.full((self.num_cells_y+2, self.num_cells_x+2), -2, dtype=object)
                else:  # for 3D
                    full_mesh = np.full((self.num_cells_z, self.num_cells_y, self.num_cells_x), -1, dtype=object)
                    aug_full_mesh = np.full((self.num_cells_z+2, self.num_cells_y+2, self.num_cells_x+2), -2, dtype=object)

                block_count = np.zeros((self.num_elems, len(candidates)))
                fill_sequence = np.zeros(self.num_cells)

                self._growth_attempt(
                    full_mesh,
                    aug_full_mesh,
                    block_count,
                    fill_sequence,
                    aug_candidates_encoded,
                    aug_candidates_encoded_ref,
                    aug_frequency_hints,
                    periodic,
                    rules,
                    rotation_table,
                    special_rules
                )
                admissible = True
                break
            except ValueError:
                print(f"Try {n_try+1}: No admissible blocks available.")

        comp_time = time.time() - start_time
        print(f"Computational time: {comp_time:.3g} s / {comp_time/3600:.3g} hrs")

        if not admissible:
            print("No successful arrangement found.")
            return

        # Post-processing
        self._post_process(
            m,
            v_array,
            full_mesh,
            block_count,
            aug_candidates,
            candidates,
            frequency_hints,
            fill_sequence,
            print_frequency,
            make_figure,
            make_gif,
            color,
            fig_name,
            save_path,
            save_mesh,
            save_mesh_path,
            save_mesh_name,
            gif_name
        )

    def _growth_attempt(
        self,
        full_mesh,
        aug_full_mesh,
        block_count,
        fill_sequence,
        aug_candidates_encoded,
        aug_candidates_encoded_ref,
        aug_frequency_hints,
        periodic,
        rules,
        rotation_table,
        special_rules
    ):
        """
        Core loop for filling all cells. Raises ValueError if no blocks can be placed.
        """
        for n_iter in tqdm(range(self.num_cells)):
            # Update the information of mesh and probabilities
            # -------------------------------------------------------------
            # Update the augmented mesh
            if self.dim == 2:
                aug_full_mesh[1:-1, 1:-1] = full_mesh
            else:  # 3D
                aug_full_mesh[1:-1, 1:-1, 1:-1] = full_mesh

            # Apply periodic constraints if needed
            if periodic:
                self._apply_periodic_boundary(full_mesh, aug_full_mesh)

            # Find neighbor blocks for all cells
            if self.dim == 2:
                left_blocks = aug_full_mesh[1:-1, :-2].flatten()
                right_blocks = aug_full_mesh[1:-1, 2:].flatten()
                top_blocks = aug_full_mesh[:-2, 1:-1].flatten()
                bottom_blocks = aug_full_mesh[2:, 1:-1].flatten()
            else:
                left_blocks = aug_full_mesh[1:-1, 1:-1, :-2].flatten()
                right_blocks = aug_full_mesh[1:-1, 1:-1, 2:].flatten()
                front_blocks = aug_full_mesh[1:-1, 2:, 1:-1].flatten()
                back_blocks = aug_full_mesh[1:-1, :-2, 1:-1].flatten()
                top_blocks = aug_full_mesh[2:, 1:-1, 1:-1].flatten()
                bottom_blocks = aug_full_mesh[:-2, 1:-1, 1:-1].flatten()

            # Update probabilities of candidate blocks
            # (x0+(n-x)*p)/n=p0 => p=(p0*n-x0)/(n-x)
            probs = (
                aug_frequency_hints * self.num_cells_elem - block_count
            ) / (
                self.num_cells_elem - np.sum(block_count, axis=1).reshape(-1, 1) + 1e-6
            )
            probs[probs <= 0] = 1e-6
            probs /= np.sum(probs, axis=1).reshape(-1, 1)
            probs = np.hstack((probs, np.zeros((self.num_elems, 1))))

            # Determine the target cell
            # -------------------------------------------------------------
            # Find cells to check
            unfilled_cells = np.argwhere(full_mesh.flatten() == -1)[:, 0]

            # Identify "remote" cells
            if self.dim == 2:
                remote_cells = np.argwhere(
                    ((left_blocks == -1) | (left_blocks == -2))
                    & ((right_blocks == -1) | (right_blocks == -2))
                    & ((top_blocks == -1) | (top_blocks == -2))
                    & ((bottom_blocks == -1) | (bottom_blocks == -2))
                )[:, 0]
            else:
                remote_cells = np.argwhere(
                    ((left_blocks == -1) | (left_blocks == -2))
                    & ((right_blocks == -1) | (right_blocks == -2))
                    & ((front_blocks == -1) | (front_blocks == -2))
                    & ((back_blocks == -1) | (back_blocks == -2))
                    & ((top_blocks == -1) | (top_blocks == -2))
                    & ((bottom_blocks == -1) | (bottom_blocks == -2))
                )[:, 0]

            if n_iter == 0:
                checked_cells = unfilled_cells
            else:
                checked_cells = np.setdiff1d(unfilled_cells, remote_cells)

            # If no cells can be placed, throw error
            if checked_cells.size == 0:
                raise ValueError("No available cells to place blocks.")

            checked_elems = self.all_elems[checked_cells]

            # Find admissible_blocks
            if self.dim == 2:
                admissible_blocks = find_admissible_blocks_2d(
                    rules, rotation_table, aug_candidates_encoded, special_rules,
                    left_blocks[checked_cells],
                    right_blocks[checked_cells],
                    top_blocks[checked_cells],
                    bottom_blocks[checked_cells]
                )
            else:
                # 3D
                # ...
                pass  # find_admissible_blocks_3d logic

            idx_map = find_indices(aug_candidates_encoded_ref, admissible_blocks.flatten())
            idx_map = idx_map.reshape(admissible_blocks.shape)

            # Probability subset
            admissible_probs = np.take_along_axis(probs[checked_elems], idx_map, 1)
            row_sums = np.sum(admissible_probs, axis=1)
            row_sums[row_sums == 0] = 1e-6
            admissible_probs /= row_sums.reshape(-1, 1)

            # Compute the entropies
            temp_ = admissible_probs.copy()
            temp_[np.isclose(temp_, 0)] = 1
            entropies = -np.einsum("ij,ij->i", admissible_probs, np.log10(temp_))

            # Determine the target cell to be filled with a block
            min_ent = entropies.min()
            candidate_ids = np.argwhere(entropies == min_ent)[:, 0]
            target_cell = np.random.choice(candidate_ids)
            target_cell = checked_cells[target_cell]
            fill_sequence[n_iter] = target_cell
            target_element = self.all_elems[target_cell]

            # Determine the target block
            # -------------------------------------------------------------
            # Find admissible_blocks
            local_id = np.argwhere(checked_cells == target_cell)[0, 0]
            target_adm_blocks = admissible_blocks[local_id]
            target_adm_probs = admissible_probs[local_id]

            rev_cumsum = np.flip(np.cumsum(target_adm_probs))
            chosen_idx = target_adm_probs.size - np.argmax(rev_cumsum < np.random.rand(1))
            if chosen_idx == target_adm_probs.size:
                chosen_idx = 0
            target_block = target_adm_blocks[chosen_idx]

            # Place block
            if self.dim == 2:
                yy = self.y_cell[target_cell]
                xx = self.x_cell[target_cell]
                full_mesh[yy, xx] = target_block
            else:
                # 3D logic
                pass

            # Update count
            chosen_block_index = np.argwhere(aug_candidates_encoded_ref == target_block)[0, 0]
            block_count[target_element, chosen_block_index] += 1

    def _apply_periodic_boundary(self, full_mesh, aug_full_mesh):
        """
        Enforce periodic constraints by copying boundary blocks into aug_full_mesh edges.
        """
        if self.dim == 2:
            aug_full_mesh[1:-1, 0] = full_mesh[:, -1]
            aug_full_mesh[1:-1, -1] = full_mesh[:, 0]
            aug_full_mesh[0, 1:-1] = full_mesh[-1, :]
            aug_full_mesh[-1, 1:-1] = full_mesh[0, :]
        else:
            aug_full_mesh[1:-1, 1:-1, 0] = full_mesh[:, :, -1]
            aug_full_mesh[1:-1, 1:-1, -1] = full_mesh[:, :, 0]
            aug_full_mesh[1:-1, -1, 1:-1] = full_mesh[:, 0, :]
            aug_full_mesh[1:-1, 0, 1:-1] = full_mesh[:, -1, :]
            aug_full_mesh[-1, 1:-1, 1:-1] = full_mesh[0, :, :]
            aug_full_mesh[0, 1:-1, 1:-1] = full_mesh[-1, :, :]

    def _post_process(
        self,
        m,
        v_array,
        full_mesh,
        block_count,
        aug_candidates,
        candidates,
        frequency_hints,
        fill_sequence,
        print_frequency,
        make_figure,
        make_gif,
        color,
        fig_name,
        save_path,
        save_mesh,
        save_mesh_path,
        save_mesh_name,
        gif_name
    ):
        """
        Computes final frequency distribution, decodes the mesh, and does optional plotting and GIF creation.
        """
        final_frequency = compute_final_frequency(block_count, self.num_elems, aug_candidates, candidates)


        if (make_figure or make_gif) and save_path != "" and not os.path.exists(save_path):
            os.makedirs(save_path)

        if save_mesh and save_mesh_path != "" and not os.path.exists(save_mesh_path):
            os.makedirs(save_mesh_path)

        if print_frequency:
            print("Input frequency hints of candidate blocks:")
            with np.printoptions(precision=4):
                print(frequency_hints)
            print("Final frequency distribution of candidate blocks:")
            with np.printoptions(precision=4):
                print(final_frequency)

        freq_error = np.linalg.norm(final_frequency - frequency_hints, 2) / self.num_elems
        print(f"2-norm error of frequency distribution: {freq_error:.3g}")

        # Decode the mesh
        idx = full_mesh.flatten().astype(int)
        full_mesh = self.names[idx].reshape(full_mesh.shape)
        if save_mesh:
            np.save(save_mesh_path + save_mesh_name, full_mesh)

        # If you want to produce figures or GIF, call plot functions
        if make_figure:
            # Example 2D call:
            elements, cell_types, nodes, element_count = plot_microstructure_2d(
                m, full_mesh, self.all_elems, self.block_library, v_array, color=color, save_path=save_path, fig_name=fig_name)

        if make_gif:
            plot_microstructure_gif()  # 需要修改！！！！！

    def _prepare_dimensions(self, mesh_size, elem_size):
        """
        Compute self.dim, self.num_cells_x/y/z, self.num_elems, etc. for 2D or 3D.
        """
        self.dim = len(mesh_size)
        self.num_elems = np.prod(mesh_size)
        self.num_cells_elem = np.prod(elem_size)
        self.num_cells = self.num_elems * self.num_cells_elem

        # Compute numbers of all cells and elements in each direction
        if self.dim == 2:
            self.num_cells_y = mesh_size[0] * elem_size[0]
            self.num_cells_x = mesh_size[1] * elem_size[1]
            all_cells = np.arange(self.num_cells)
            y_cell, x_cell = np.divmod(all_cells, self.num_cells_x)
            self.y_cell = y_cell
            self.x_cell = x_cell
            y_elem, _ = np.divmod(y_cell, elem_size[0])
            x_elem, _ = np.divmod(x_cell, elem_size[1])
            self.all_elems = y_elem * mesh_size[1] + x_elem
        else:
            self.num_cells_z = mesh_size[0] * elem_size[0]
            self.num_cells_y = mesh_size[1] * elem_size[1]
            self.num_cells_x = mesh_size[2] * elem_size[2]
            all_cells = np.arange(self.num_cells)
            z_cell, temp = np.divmod(all_cells, self.num_cells_x * self.num_cells_y)
            y_cell, x_cell = np.divmod(temp, self.num_cells_x)
            self.z_cell = z_cell
            self.y_cell = y_cell
            self.x_cell = x_cell
            z_elem, _ = np.divmod(z_cell, elem_size[0])
            y_elem, _ = np.divmod(y_cell, elem_size[1])
            x_elem, _ = np.divmod(x_cell, elem_size[2])
            self.all_elems = (
                z_elem * mesh_size[1] * mesh_size[2]
                + y_elem * mesh_size[2]
                + x_elem
            )

    def _add_unfilled_and_wall(self, rules, aug_candidates_encoded, rotation_table):
        """
        Add cases of "unfilled" and "wall" to rules and rotation_table.
        """
        if aug_candidates_encoded.size > rules.shape[1]:
            rules = np.hstack((rules, np.ones(
                (rules.shape[0], aug_candidates_encoded.size-rules.shape[1]), dtype=int) * -1
            ))
        elif aug_candidates_encoded.size < rules.shape[1]:
            aug_candidates_encoded = np.hstack((
                aug_candidates_encoded,
                np.ones(rules.shape[1]-aug_candidates_encoded.size, dtype=int) * -1,
            ))
        rules = np.vstack((rules, aug_candidates_encoded, aug_candidates_encoded))

        rotation_table = np.vstack((
            rotation_table,
            np.ones(rotation_table.shape[1], dtype=int) * -2,  # Wall
            np.ones(rotation_table.shape[1], dtype=int) * -1,  # Unfilled
        ))
        return rules, aug_candidates_encoded, rotation_table
