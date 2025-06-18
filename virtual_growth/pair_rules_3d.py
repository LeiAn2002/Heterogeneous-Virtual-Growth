# pair_rules_3d.py
import numpy as np
from utils.array_list_operations import find_indices, find_max_length_nested_list
from collections import defaultdict
from typing import Dict


class PairRules3D:
    """
    Build 3-D pair-adjacency rules / rotation-table / special-rules
    in an OOP style (mirrors PairRules2D).
    """

    # ---------- public API --------------------------------------------------

    def __init__(self, block_library):
        self.block_library = block_library     # an existing BlockLibrary instance

    def generate_rules(self, block_names: list):
        """
        Main entry. Given a list of block names, return

            (unique_blocks,
             extended_names,
             encoded_rotation_table,
             encoded_adj_rules,
             encoded_special_rules,
             uid2oid)

        No file I/O is performed here.
        """
        all_unique_blocks = []
        all_block_names = []
        all_extended_block_names = []
        uid2oid: Dict[str, Dict[int, int]] = defaultdict(dict)

        # 1. generate all distinct rotations for each block
        for name in block_names:
            blk_obj = self.block_library.create_block(name)
            ref_matrix = blk_obj.get_adjacent_matrix()           # (Z, Y, X)
            rotations = self._rotate_block(ref_matrix)          # 24 variants
            uniq_idx = self._remove_repeated_blocks(rotations)

            for count, idx in enumerate(uniq_idx):
                all_unique_blocks.append(rotations[idx])
                all_block_names.append(name)
                all_extended_block_names.append(f"{name} {count}")
                uid2oid[name][count] = int(idx)

        # 2. pairwise admissible rule list
        rules = []
        for L_blk, L_name in zip(all_unique_blocks, all_extended_block_names):
            for R_blk, R_name in zip(all_unique_blocks, all_extended_block_names):
                if self._admissible_pairs(L_blk, R_blk):
                    rules.append([L_name, R_name])

        # 3. rotation table
        rot_table = self._generate_rotation_table(all_unique_blocks, all_extended_block_names)

        # 4. special placement rules (eight corners of the mesh)
        special = self._generate_special_rules(all_unique_blocks, all_extended_block_names)

        # 5. encode to int arrays
        enc_rot, enc_rules, enc_spec = self._encode(
            all_extended_block_names, rot_table, rules, special
        )

        return (
            all_unique_blocks,
            all_extended_block_names,
            enc_rot,
            enc_rules,
            enc_spec,
            uid2oid
        )

    # ---------- helpers -----------------------------------------------------

    @staticmethod
    def _rotate_block(mat):
        """
        Enumerate all 24 orientation variants in 3-D:

        * Keep the original “front” face, rotate around Z axis 0-3 times.
        * Rotate the original block so that the old front faces become
          top / bottom / left / right / back, then for each of those
          orientations rotate around the new Z (old axes) 0-3 times.

        Returns an array with shape (24, Z, Y, X).
        """
        mats = []
        # front remains front (Y axis still points forward)
        for i in range(4):
            mats.append(np.rot90(mat, i, axes=(1, 2)))
        # front -> top
        top = np.rot90(mat, -1, axes=(0, 1))
        for i in range(4):
            mats.append(np.rot90(top, i, axes=(0, 2)))
        # front -> bottom
        bottom = np.rot90(mat, 1, axes=(0, 1))
        for i in range(4):
            mats.append(np.rot90(bottom, i, axes=(0, 2)))
        # front -> left
        left = np.rot90(mat, 1, axes=(0, 2))
        for i in range(4):
            mats.append(np.rot90(left, i, axes=(0, 1)))
        # front -> right
        right = np.rot90(mat, -1, axes=(0, 2))
        for i in range(4):
            mats.append(np.rot90(right, i, axes=(0, 1)))
        # front -> back
        back = np.rot90(mat, 2, axes=(0, 1))
        for i in range(4):
            mats.append(np.rot90(back, i, axes=(1, 2)))

        return np.array(mats, dtype=int)

    @staticmethod
    def _remove_repeated_blocks(blocks):
        """Return indices of unique blocks (value equality)."""
        _, idx = np.unique(blocks, axis=0, return_index=True)
        return np.sort(idx)

    # --- admissible check ---------------------------------------------------

    @staticmethod
    def _is_detached(f1, f2):
        return np.sum(f1 + f2) == 0

    @staticmethod
    def _is_connected(f1, f2):
        return np.sum(f1 * f2) > 0

    def _admissible_pairs(self, L, R):
        """
        Apply the 3-D admissibility logic:

        * faces must be fully connected or fully detached
        * forbids “corner-to-corner” and “line-to-line” contact patterns
        (porting of the original procedural rules).
        """
        # alias faces
        Lx0, Lx1 = L[:, :, 0], L[:, :, -1]
        Rx0, Rx1 = R[:, :, 0], R[:, :, -1]
        Ly0, Ly1 = L[:, 0, :], L[:, -1, :]
        Ry0, Ry1 = R[:, 0, :], R[:, -1, :]
        Lz0, Lz1 = L[0, :, :], L[-1, :, :]
        Rz0, Rz1 = R[0, :, :], R[-1, :, :]

        # basic fully-connect / fully-detach between facing X-faces
        flag1 = self._is_detached(Lx1, Rx0) or self._is_connected(Lx1, Rx0)

        # forbid two corner blocks facing each other
        edge1 = self._is_detached(Lx0, Rx1)
        edge2 = self._is_connected(Lx1, Rx0)
        edge3 = self._is_detached(Lz0, Rz0)
        edge4 = self._is_detached(Ly0, Ry0)
        edge5 = self._is_detached(Lz1, Rz1)
        edge6 = self._is_detached(Ly1, Ry1)
        edge7 = self._is_connected(Lz0, Rz0)
        edge8 = self._is_connected(Ly0, Ry0)
        edge9 = self._is_connected(Lz1, Rz1)
        edge10 = self._is_connected(Ly1, Ry1)
        flag2 = edge1 & edge2 & (
            (edge3 & edge4 & (edge9 | edge10)) |
            (edge4 & edge5 & (edge10 | edge7)) |
            (edge5 & edge6 & (edge7 | edge8)) |
            (edge6 & edge3 & (edge8 | edge9))
        )

        # forbid connecting two complete straight lines
        e1 = self._is_detached(Lz0, Rz0)
        e2 = self._is_detached(Ly0, Ry0)
        e3 = self._is_detached(Lz1, Rz1)
        e4 = self._is_detached(Ly1, Ry1)
        e5 = self._is_connected(Lx0, Rx1)
        e6 = self._is_connected(Lx1, Rx0)
        flag3 = e1 & e2 & e3 & e4 & e5 & e6

        return flag1 and (not flag2) and (not flag3)

    # --- rotation table & special rules ------------------------------------

    def _generate_rotation_table(self, blocks, block_names):
        """
        Return a dict {block_name: [rot1, rot2, rot3, rot4, rot5]}.
        """
        def find_block_name(block, blocks, block_names):
            """Find the block name of a given block."""
            temp = np.sum((blocks - block)**2, axis=(1, 2, 3))
            index = np.argwhere(temp == 0)[0, 0]
            return block_names[index], index

        rotation_table = np.empty((len(block_names), 6), dtype=object)
        for n, block in enumerate(blocks):
            # Rotation in the x-y plane
            for i in range(4):
                rotated_block = np.rot90(block, i, axes=(1, 2))
                rotated_name, _ = find_block_name(rotated_block, blocks, block_names)
                rotation_table[n][i] = rotated_name

            # Rotation from front to top
            rotated_block = np.rot90(block, -1, axes=(0, 1))
            rotated_name, _ = find_block_name(rotated_block, blocks, block_names)
            rotation_table[n][i+1] = rotated_name

            # Rotatio from front to bottom
            rotated_block = np.rot90(block, 1, axes=(0, 1))
            rotated_name, _ = find_block_name(rotated_block, blocks, block_names)
            rotation_table[n][i+2] = rotated_name

        # Convert the np.ndarray to a dictionary
        rotation_dict = {}
        for n in range(len(block_names)):
            rotation_dict[rotation_table[n, 0]] = rotation_table[n, 1:].tolist()

        return rotation_dict

    def _generate_special_rules(self, blocks, names):
        """
        Identify blocks that cannot sit at any of the 8 mesh corners.
        Corner code legend:
          111 → (+X,+Y,+Z) etc.
        """
        codes = [
            "111", "-111", "-1-11", "1-11",
            "11-1", "-11-1", "-1-1-1", "1-1-1"
        ]
        special = {c: [] for c in codes}

        for blk, n in zip(blocks, names):
            if np.sum(blk[0, :, :] + blk[:, 0, :] + blk[:, :, -1]) == 0:
                special["111"].append(n)
            if np.sum(blk[0, :, :] + blk[:, 0, :] + blk[:, :, 0]) == 0:
                special["-111"].append(n)
            if np.sum(blk[0, :, :] + blk[:, -1, :] + blk[:, :, 0]) == 0:
                special["-1-11"].append(n)
            if np.sum(blk[0, :, :] + blk[:, -1, :] + blk[:, :, -1]) == 0:
                special["1-11"].append(n)
            if np.sum(blk[-1, :, :] + blk[:, 0, :] + blk[:, :, -1]) == 0:
                special["11-1"].append(n)
            if np.sum(blk[-1, :, :] + blk[:, 0, :] + blk[:, :, 0]) == 0:
                special["-11-1"].append(n)
            if np.sum(blk[-1, :, :] + blk[:, -1, :] + blk[:, :, 0]) == 0:
                special["-1-1-1"].append(n)
            if np.sum(blk[-1, :, :] + blk[:, -1, :] + blk[:, :, -1]) == 0:
                special["1-1-1"].append(n)

        return special

    # --- encode -------------------------------------------------------------

    @staticmethod
    def _encode(names, rotation_table, inp_rules, special_rules):
        """Encode strings to numbers."""
        # Encode the rotation table
        names = np.array(names)
        num_rows = len(rotation_table.keys())
        num_cols = find_max_length_nested_list(rotation_table.values())
        encoded_rotation_table = np.zeros((num_rows, num_cols), dtype=int)
        for key, value in rotation_table.items():
            row = np.argwhere(names == key)[0, 0]
            indices = find_indices(names, value)
            encoded_rotation_table[row] = indices

        # Encode the adjacency rule
        rules = {}
        for (key, val) in inp_rules:
            if key not in rules.keys():
                rules[key] = [val]
            else:
                rules[key].append(val)
        num_rows = len(rules.keys())
        num_cols = find_max_length_nested_list(rules.values())
        encoded_rules = np.zeros((num_rows, num_cols), dtype=int)
        for key, value in rules.items():
            row = np.argwhere(names == key)[0, 0]
            indices = find_indices(names, value)
            indices = np.hstack((indices, np.ones(num_cols-indices.size, dtype=int)*-1))
            encoded_rules[row] = indices

        # Encode special rules
        num_rows = len(special_rules.keys())
        num_cols = find_max_length_nested_list(special_rules.values())
        encoded_special_rules = np.zeros((num_rows, num_cols), dtype=int)
        for n, value in enumerate(special_rules.values()):
            encoded_special_rules[n] = find_indices(names, value)

        return encoded_rotation_table, encoded_rules, encoded_special_rules
