from abc import ABC, abstractmethod
import numpy as np
from utils.nodes_operations import arc_points, rotate_nodes
from utils.create_rectangle_mesh import create_rectangle_mesh
from utils.mesh_operations import process_raw_mesh, coord_transformation

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
    def generate_nodes(self, m, thickness_matrix, v, rotation, **kwargs):
        pass

    @abstractmethod
    def generate_elements(self, **kwargs):
        pass

    @abstractmethod
    def generate_mesh(self, m, thickness_matrix, v, **kwargs):
        pass


@register_block_class
class CircleBlock2D(Block):

    type_name = "circle"

    def __init__(self, m, v, rotation, discre_number=5):
        self.m = m
        self.v = v
        self.rotation = rotation
        self.discre_number = discre_number

    def get_adjacent_matrix(self, **kwargs):
        adj_matrix = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]).astype(int),
        return adj_matrix

    def get_thickness(self, **kwargs):
        thickness = self.m * self.v
        thickness_matrix = np.array([
            [thickness, thickness],
            [thickness, thickness]
        ])  # thickness of 4 edges, order: left, bottom, right, top
        rotated_thickness_matrix = np.roll(thickness_matrix, self.rotation)
        return rotated_thickness_matrix

    def generate_nodes(self, thickness_matrix, **kwargs):
        scale = 1 - self.v
        t = np.roll(thickness_matrix, -self.rotation)  # original thickness_matrix
        outside_nodes = np.array([
            [-self.m, -t[0, 0], 0],
            [-t[0, 1], -self.m, 0],
            [t[0, 1], -self.m, 0],
            [self.m, -t[1, 0], 0],
            [self.m, t[1, 0], 0],
            [t[1, 1], self.m, 0],
            [-t[1, 1], self.m, 0],
            [-self.m, t[0, 0], 0]
        ])
        inside_nodes = np.array([
            [-self.m*scale, -t[0, 0]*scale, 0],
            [-t[0, 1]*scale, -self.m*scale, 0],
            [t[0, 1]*scale, -self.m*scale, 0],
            [self.m*scale, -t[1, 0]*scale, 0],
            [self.m*scale, t[1, 0]*scale, 0],
            [t[1, 1]*scale, self.m*scale, 0],
            [-t[1, 1]*scale, self.m*scale, 0],
            [-self.m*scale, t[0, 0]*scale, 0]
        ])

        rotated_outside_nodes = rotate_nodes(outside_nodes, self.rotation)
        rotated_inside_nodes = rotate_nodes(inside_nodes, self.rotation)

        nodes_list = []

        for i in range(4):
            nodes_list += arc_points(rotated_inside_nodes[2*i, :2], rotated_inside_nodes[2*i+1, :2], self.discre_number)

        for i in range(4):
            nodes_list += arc_points(rotated_outside_nodes[2*i, :2], rotated_outside_nodes[2*i+1, :2], self.discre_number)

        nodes = np.array(nodes_list).astype(float)
        return nodes

    def generate_elements(self, **kwargs):
        elements = np.array(
            [[4, i, 4 * self.discre_number + i, 4 * self.discre_number + 1 + i, i + 1] for i in range(4 * self.discre_number - 1)] +
            [[4, 4 * self.discre_number-1, 8*self.discre_number-1, 4*self.discre_number, 0]]
        )
        return elements

    def generate_mesh(self, thickness_matrix, num_elems_d, **kwargs):
        all_nodes = []
        all_elements = []
        point_sets = self.generate_nodes(self.m, thickness_matrix, self.v, self.rotation)
        # part 1
        for i in range(self.discre_number-1):
            nodes, elements = create_rectangle_mesh(2, 2, num_elems_d, 1)
            nodes -= 1
            vx = np.array([point_sets[4*self.discre_number+1+i][0], point_sets[i+1][0], point_sets[i][0], point_sets[4*self.discre_number+i][0]])
            vy = np.array([point_sets[4*self.discre_number+1+i][1], point_sets[i+1][1], point_sets[i][1], point_sets[4*self.discre_number+i][1]])
            nodes[:, 0], nodes[:, 1] = coord_transformation(
                nodes[:, 0], nodes[:, 1], vx, vy)
            all_nodes.append(nodes)
            all_elements.append(elements)

        special_nodes_1, special_elements_1 = create_rectangle_mesh(2, 2, num_elems_d, num_elems_d)
        special_nodes_1 -= 1
        vx = np.array([point_sets[5*self.discre_number][0], point_sets[self.discre_number][0], point_sets[self.discre_number-1][0], point_sets[5*self.discre_number-1][0]])
        vy = np.array([point_sets[5*self.discre_number][1], point_sets[self.discre_number][1], point_sets[self.discre_number-1][1], point_sets[5*self.discre_number-1][1]])
        special_nodes_1[:, 0], special_nodes_1[:, 1] = coord_transformation(
            special_nodes_1[:, 0], special_nodes_1[:, 1], vx, vy)
        all_nodes.append(special_nodes_1)
        all_elements.append(special_elements_1)

        # part 2
        for i in range(self.discre_number, 2*self.discre_number-1):
            nodes, elements = create_rectangle_mesh(2, 2, num_elems_d, 1)
            nodes -= 1
            vx = np.array([point_sets[4*self.discre_number+1+i][0], point_sets[i+1][0], point_sets[i][0], point_sets[4*self.discre_number+i][0]])
            vy = np.array([point_sets[4*self.discre_number+1+i][1], point_sets[i+1][1], point_sets[i][1], point_sets[4*self.discre_number+i][1]])
            nodes[:, 0], nodes[:, 1] = coord_transformation(
                nodes[:, 0], nodes[:, 1], vx, vy)
            all_nodes.append(nodes)
            all_elements.append(elements)

        special_nodes_2, special_elements_2 = create_rectangle_mesh(2, 2, num_elems_d, num_elems_d)
        special_nodes_2 -= 1
        vx = np.array([point_sets[6*self.discre_number][0], point_sets[2*self.discre_number][0], point_sets[2*self.discre_number-1][0], point_sets[6*self.discre_number-1][0]])
        vy = np.array([point_sets[6*self.discre_number][1], point_sets[2*self.discre_number][1], point_sets[2*self.discre_number-1][1], point_sets[6*self.discre_number-1][1]])
        special_nodes_2[:, 0], special_nodes_2[:, 1] = coord_transformation(
            special_nodes_2[:, 0], special_nodes_2[:, 1], vx, vy)
        all_nodes.append(special_nodes_2)
        all_elements.append(special_elements_2)

        # part 3
        for i in range(2*self.discre_number, 3*self.discre_number-1):
            nodes, elements = create_rectangle_mesh(2, 2, num_elems_d, 1)
            nodes -= 1
            vx = np.array([point_sets[4*self.discre_number+1+i][0], point_sets[i+1][0], point_sets[i][0], point_sets[4*self.discre_number+i][0]])
            vy = np.array([point_sets[4*self.discre_number+1+i][1], point_sets[i+1][1], point_sets[i][1], point_sets[4*self.discre_number+i][1]])
            nodes[:, 0], nodes[:, 1] = coord_transformation(
                nodes[:, 0], nodes[:, 1], vx, vy)
            all_nodes.append(nodes)
            all_elements.append(elements)

        special_nodes_3, special_elements_3 = create_rectangle_mesh(2, 2, num_elems_d, num_elems_d)
        special_nodes_3 -= 1
        vx = np.array([point_sets[7*self.discre_number][0], point_sets[3*self.discre_number][0], point_sets[3*self.discre_number-1][0], point_sets[7*self.discre_number-1][0]])
        vy = np.array([point_sets[7*self.discre_number][1], point_sets[3*self.discre_number][1], point_sets[3*self.discre_number-1][1], point_sets[7*self.discre_number-1][1]])
        special_nodes_3[:, 0], special_nodes_3[:, 1] = coord_transformation(
            special_nodes_3[:, 0], special_nodes_3[:, 1], vx, vy)
        all_nodes.append(special_nodes_3)
        all_elements.append(special_elements_3)

        # part 4
        for i in range(3*self.discre_number, 4*self.discre_number-1):
            nodes, elements = create_rectangle_mesh(2, 2, num_elems_d, 1)
            nodes -= 1
            vx = np.array([point_sets[4*self.discre_number+1+i][0], point_sets[i+1][0], point_sets[i][0], point_sets[4*self.discre_number+i][0]])
            vy = np.array([point_sets[4*self.discre_number+1+i][1], point_sets[i+1][1], point_sets[i][1], point_sets[4*self.discre_number+i][1]])
            nodes[:, 0], nodes[:, 1] = coord_transformation(
                nodes[:, 0], nodes[:, 1], vx, vy)
            all_nodes.append(nodes)
            all_elements.append(elements)

        special_nodes_4, special_elements_4 = create_rectangle_mesh(2, 2, num_elems_d, num_elems_d)
        special_nodes_4 -= 1
        vx = np.array([point_sets[4*self.discre_number][0], point_sets[0][0], point_sets[4*self.discre_number-1][0], point_sets[8*self.discre_number-1][0]])
        vy = np.array([point_sets[4*self.discre_number][1], point_sets[0][1], point_sets[4*self.discre_number-1][1], point_sets[8*self.discre_number-1][1]])
        special_nodes_4[:, 0], special_nodes_4[:, 1] = coord_transformation(
            special_nodes_4[:, 0], special_nodes_4[:, 1], vx, vy)
        all_nodes.append(special_nodes_4)
        all_elements.append(special_elements_4)

        elements, cell_types, nodes = process_raw_mesh(
            all_nodes, all_elements)
        return elements, cell_types, nodes


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
