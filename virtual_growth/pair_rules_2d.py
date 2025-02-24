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
from utils.array_list_operations import find_indices, find_max_length_nested_list


class PairRules2D:
    """
    Used for batch generating various rotated variants of Blocks and
    constructing Pair Rules in a 2D setting.

    Example usage:
    1) Pass in a `block_library`.
    2) Instantiate this class: `PairRules2D(block_library)`.
    3) Call `generate_rules(block_names, v_array, m, ...)` to generate
       and return the corresponding results.
    """

    def __init__(self, block_library):
        self.block_library = block_library

    def generate_rules(self, block_names: list):
        """
        Generates the rules required for virtual growth, including:
        - Unique blocks (removing duplicates after rotation)
        - Pair adjacency rules
        - Rotation table
        - Special rules
        - Encoding of these rules

        Returns:
        --------
        (all_unique_blocks, all_extended_block_names,
        encoded_rotation_table, encoded_rules, encoded_special_rules)
        """
        all_unique_blocks = []
        all_block_names = []
        all_extended_block_names = []

        for block_name in block_names:
            block_obj = self._get_block_from_library(block_name)

            input_block = block_obj.get_adjacent_matrix()
            out_blocks = self._rotate_block(input_block)
            unique_indices = self._remove_repeated_blocks(out_blocks)

            for count, i_idx in enumerate(unique_indices):
                all_unique_blocks.append(out_blocks[i_idx])
                all_block_names.append(block_name)
                all_extended_block_names.append(f"{block_name} {count}")

        # 2) pair adjacency rules
        rules = []
        for (left_block, left_name) in zip(all_unique_blocks, all_extended_block_names):
            for (right_block, right_name) in zip(all_unique_blocks, all_extended_block_names):
                flag = self._admissible_pairs(left_block, right_block)
                if flag:
                    rules.append([left_name, right_name])

        # 3) rotation table
        rotation_table = self._generate_rotation_table(all_unique_blocks, all_extended_block_names)

        # 4) special rules
        special_rules = self._generate_special_rules(all_unique_blocks, all_extended_block_names)

        # 5) encode
        (
            encoded_rotation_table,
            encoded_rules,
            encoded_special_rules
        ) = self._encode(all_extended_block_names, rotation_table, rules, special_rules)

        return (
            all_unique_blocks,
            all_extended_block_names,
            encoded_rotation_table,
            encoded_rules,
            encoded_special_rules,
        )

    def _get_block_from_library(self, block_name):
        """Get the block object from the block library."""
        return self.block_library.create_block(block_name)

    @staticmethod
    def _rotate_block(inp_block):
        """
        Rotate the block to find the variations. There are 4 possibilities.
        y
        ↑
        O → x
        """
        inp_block = np.array(inp_block)
        out_blocks = np.zeros((4, *inp_block.shape))  # At most 4 variations

        for i in range(4):  # Rotation in the x-y plane
            out_blocks[i] = np.rot90(inp_block, i, axes=(0, 1))

        return out_blocks.astype(int)

    @staticmethod
    def _remove_repeated_blocks(inp_blocks):
        """Remove repeated blocks."""
        _, unique_indices = np.unique(inp_blocks, axis=0, return_index=True)
        return np.sort(unique_indices)

    @staticmethod
    def _admissible_pairs(left, right):
        """Check if two blocks are admissible or not."""

        def isdetached(edge1, edge2):
            """Check if two edges are detached or not."""
            return np.sum(edge1 + edge2) == 0

        def isconnected(edge1, edge2):
            """Check if two edges are connected or not."""
            return edge1 @ edge2 > 0

        # Basic requirement: fully connected or fully detached
        flag1 = isdetached(left[:, -1], right[:, 0]) or isconnected(left[:, -1], right[:, 0])

        # Two corner-shaped blocks cannot be placed face to face (e.g., ┏ ┓)
        index1 = isdetached(left[:, 0], right[:, -1])
        index2 = isconnected(left[:, -1], right[:, 0])
        index3 = isdetached(left[0, :], right[0, :])
        index4 = isconnected(left[-1, :], right[-1, :])
        index5 = isdetached(left[-1, :], right[-1, :])
        index6 = isconnected(left[0, :], right[0, :])
        flag2 = index1 and index2 and ((index3 and index4) or (index5 and index6))

        # Two lines cannot be connected (e.g., — —)
        index1 = isdetached(left[0, :], right[0, :])
        index2 = isdetached(left[-1, :], right[-1, :])
        index3 = isconnected(left[:, 0], right[:, -1])
        index4 = isconnected(left[:, -1], right[:, 0])
        flag3 = index1 and index2 and index3 and index4

        return flag1 and (not flag2) and (not flag3)

    def _generate_rotation_table(self, blocks, block_names):
        """
        Find the corresponding block after the rotation.
        Returns a dictionary:
        { block_name: [rotated_1, rotated_2, rotated_3], ...}
        """
        def find_block_name(block, blocks, block_names):
            """Find the block name of a given block."""
            temp = np.sum((blocks - block) ** 2, axis=(1, 2))
            index = np.argwhere(temp == 0)[0, 0]
            return block_names[index], index

        rotation_table = np.empty((len(block_names), 4), dtype=object)
        for n, block in enumerate(blocks):
            # Rotation in the x-y plane
            for i in range(4):
                rotated_block = np.rot90(block, i, axes=(0, 1))
                rotated_name, _ = find_block_name(rotated_block, blocks, block_names)
                rotation_table[n][i] = rotated_name

        # Convert the np.ndarray to a dictionary
        rotation_dict = {}
        for n in range(len(block_names)):
            original_name = rotation_table[n, 0]
            rotation_dict[original_name] = rotation_table[n, 1:].tolist()

        return rotation_dict

    def _generate_special_rules(self, blocks, names):
        """Some blocks cannot be placed at special positions of mesh."""
        # Corner-shaped blocks cannot be placed at the mesh corners.
        special_rules = {
            "11": [],
            "-11": [],
            "-1-1": [],
            "1-1": [],
        }
        for block, name in zip(blocks, names):
            if np.sum(block[:, 0] + block[-1, :]) == 0:
                special_rules["11"].append(name)
            if np.sum(block[:, -1] + block[-1, :]) == 0:
                special_rules["-11"].append(name)
            if np.sum(block[:, -1] + block[0, :]) == 0:
                special_rules["-1-1"].append(name)
            if np.sum(block[:, 0] + block[0, :]) == 0:
                special_rules["1-1"].append(name)
        return special_rules

    def _encode(names, rotation_table, inp_rules, special_rules):
        """Encode strings to numbers."""
        # Encode the rotation table
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
