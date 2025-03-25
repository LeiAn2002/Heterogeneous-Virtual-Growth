from abc import ABC, abstractmethod
import numpy as np
from blocks.random_block import block_generation
from utils.rotation_matrix import rotate_thickness_matrix

ALL_BLOCKS = {}


def register_block_class(cls):
    """
    A decorator that reads cls.type_name and registers
    cls into a global dictionary ALL_BLOCK_CLASSES.
    """
    if not hasattr(cls, "type_name"):
        raise ValueError("Block class must define a 'type_name' attribute.")
    name = cls.type_name
    ALL_BLOCKS[name] = cls
    return cls


class Block(ABC):

    @abstractmethod
    def get_adjacent_matrix(self, **kwargs):
        pass

    @abstractmethod
    def get_thickness(self, m, v, rotation, **kwargs):
        pass

    @abstractmethod
    def generate_block_shape(self, m, thickness_matrix, v, rotation, **kwargs):
        pass

    @abstractmethod
    def generate_elements(self, **kwargs):
        pass

    @abstractmethod
    def generate_mesh(self, m, thickness_matrix, v, **kwargs):
        pass


@register_block_class
class CrossBlock2D(Block):

    type_name = "cross"

    def __init__(self, m=0.75, v_range=[0.4, 0.6], rotation=0, random_radius=0.6, **kwargs):
        self.m = m
        self.v_range = v_range
        self.rotation = rotation
        self.random_radius = random_radius
        self.vertical_points = np.array([
            (0, -1),   # bottom boundary
            (0, -0.9999999),  # garantee perpendicular to the boundary

            (0, 0.9999999),   # top boundary
            (0, 1)])
        self.horizontal_points = np.array([
            (-1, 0),   # left boundary
            (-0.9999999, 0),

            (0.9999999, 0),   # right boundary
            (1, 0)])
        self.outer_basic_points = np.concatenate((self.vertical_points, self.horizontal_points))
        self.vertical_inner_points = np.array([
            (0, -1/3),
            (0, 1/3)
        ])
        self.horizontal_inner_points = np.array([
            (-1/3, 0),
            (1/3, 0)
        ])
        self.inner_basic_points = np.concatenate((self.vertical_inner_points, self.horizontal_inner_points))
        self.basic_points = np.concatenate((self.outer_basic_points, self.inner_basic_points))
        self.outer_count = len(self.outer_basic_points)

        self.curve_definitions = [
            [0, 1, 8, 9, 2, 3],   # bottom->inner->top (vertical)
            [4, 5, 10, 11, 6, 7],   # left->inner->right (horizontal)
        ]
        self.number_of_curves = len(self.curve_definitions)
        self.number_of_inner_points = len(self.vertical_inner_points)

    def get_adjacent_matrix(self, **kwargs):
        adj_matrix = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ]).astype(int)
        return adj_matrix

    def get_thickness(self, **kwargs):
        lower_boundary = self.v_range[0]
        upper_boundary = self.v_range[1]

        # thickness of 4 edges, order: bottom, top, left, right
        thickness_matrix = np.random.uniform(lower_boundary, upper_boundary, size=(2, 2))
        rotated_thickness_matrix = rotate_thickness_matrix(thickness_matrix, self.rotation)
        return rotated_thickness_matrix

    def generate_block_shape(self, thickness_matrix, **kwargs):
        t = rotate_thickness_matrix(thickness_matrix, -self.rotation)  # original thickness_matrix
        forbidden_edges = np.array([
                           [(-1, -1, -1, 1),
                            (1, -1, 1, 1),
                            (-1, -1, -t[0][0], -1),
                            (t[0][0], -1, 1, -1),
                            (-1, 1, -t[0][1], 1),
                            (t[0][1], 1, 1, 1)],
                           [(-1, -1, 1, -1),
                            (-1, 1, 1, 1),
                            (-1, -1, -1, -t[1][0]),
                            (-1, t[1][0], -1, 1),
                            (1, -1, 1, -t[1][1]),
                            (1, t[1][1], 1, 1)
                            ]])
        lower_bound = self.v_range[0]
        upper_bound = self.v_range[1]
        middle = np.random.uniform(lower_bound, upper_bound, size=(self.number_of_curves, self.number_of_inner_points))
        ones_left = np.ones((self.number_of_curves, 2))
        ones_right = np.ones((self.number_of_curves, 2))
        vf = np.hstack([ones_left, middle, ones_right])
        block = block_generation(
                basic_points=self.basic_points,
                outer_count=self.outer_count,
                r=self.random_radius,
                curve_definitions=self.curve_definitions,
                vf=vf,
                forbidden_edges_set=forbidden_edges,
                r_filter=3,
            )

        rotated_block = np.rot90(block, self.rotation)
        return rotated_block

    def generate_elements(self, **kwargs):
        pass

    def generate_mesh(self, thickness_matrix, num_elems_d, **kwargs):
        pass


@register_block_class
class LBlock2D(Block):

    type_name = "L"

    def __init__(self, m=0.75, v_range=[0.4, 0.6], rotation=0, random_radius=0.6, **kwargs):
        self.m = m
        self.v_range = v_range
        self.rotation = rotation
        self.random_radius = random_radius
        self.outer_basic_points = np.array([
            (0, 1),   # left boundary
            (0, 0.99999999),
            
            (0.9999999, 0),   # right boundary
            (1, 0),
        ])
        self.inner_basic_points = np.array([
            (0, 0.5),
            (0.5, 0)
        ])
        self.basic_points = np.concatenate((self.outer_basic_points, self.inner_basic_points))
        self.outer_count = len(self.outer_basic_points)

        self.curve_definitions = [[0, 1, 4, 5, 2, 3]]  # top->inner->right
        self.number_of_curves = len(self.curve_definitions)
        self.number_of_inner_points = len(self.inner_basic_points)

    def get_adjacent_matrix(self, **kwargs):
        adj_matrix = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype(int)
        return adj_matrix

    def get_thickness(self, **kwargs):
        lower_boundary = self.v_range[0]
        upper_boundary = self.v_range[1]

        # thickness of 4 edges, order: bottom, top, left, right
        thickness_matrix = np.array([[0, np.random.uniform(lower_boundary, upper_boundary)], [0, np.random.uniform(lower_boundary, upper_boundary)]])
        rotated_thickness_matrix = rotate_thickness_matrix(thickness_matrix, self.rotation)
        return rotated_thickness_matrix

    def generate_block_shape(self, thickness_matrix, **kwargs):
        t = rotate_thickness_matrix(thickness_matrix, -self.rotation)  # original thickness_matrix
        forbidden_edges = np.array([
                           [(-1, -1, -1, 1),
                            (-1, -1, 1, -1),
                            (-1, 1, -t[0][1], 1),
                            (t[0][1], 1, 1, 1),
                            (1, -1, 1, -t[1][1]),
                            (1, t[1][1], 1, 1)]])
        lower_bound = self.v_range[0]
        upper_bound = self.v_range[1]
        middle = np.random.uniform(lower_bound, upper_bound, size=(self.number_of_curves, self.number_of_inner_points))
        ones_left = np.ones((self.number_of_curves, 2))
        ones_right = np.ones((self.number_of_curves, 2))
        vf = np.hstack([ones_left, middle, ones_right])
        block = block_generation(
                basic_points=self.basic_points,
                outer_count=self.outer_count,
                r=self.random_radius,
                curve_definitions=self.curve_definitions,
                vf=vf,
                forbidden_edges_set=forbidden_edges,
                r_filter=3,
            )

        block = np.rot90(block, self.rotation, axes=(0, 1))
        return block

    def generate_elements(self, **kwargs):
        pass

    def generate_mesh(self, thickness_matrix, num_elems_d, **kwargs):
        pass


class BlockLibrary:

    def __init__(self):
        self._registry = {}

    def register_block(self, block_type_name: str):
        """
        Takes a string name (like 'circle') and looks up the corresponding
        block class from the global ALL_BLOCK_CLASSES dictionary.
        """
        block_cls = ALL_BLOCKS.get(block_type_name)
        if block_cls is None:
            raise ValueError(f"No block class found for {block_type_name}")
        self._registry[block_type_name] = block_cls

    def create_block(self, block_type_name, *args, **kwargs) -> Block:
        cls = self._registry.get(block_type_name)
        if cls is None:
            raise ValueError(f"Unrecognized block type: {block_type_name}")
        return cls(*args, **kwargs)
