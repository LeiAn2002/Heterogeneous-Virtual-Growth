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

from blocks.blocks import BlockLibrary
from virtual_growth.virual_growth_engine import VirtualGrowthEngine
import numpy as np


def main(
        mesh_size,
        elem_size,
        candidates,
        frequency_hints,
        v_array: np.ndarray,
        r_array: np.ndarray,
        m=6,
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

    # 1) Create a block library and register blocks
    library = BlockLibrary()
    for block_type in candidates:
        library.register_block(block_type)

    # 2) Create the engine
    engine = VirtualGrowthEngine(library)

    # 3) Generate pair rules
    engine.generate_pair_rules(candidates)

    # 4) Run the virtual growth
    engine.run_growth(
        mesh_size,
        elem_size,
        candidates,
        frequency_hints,
        v_array,
        r_array,
        m,
        periodic=periodic,
        num_tries=num_tries,
        print_frequency=print_frequency,
        make_figure=make_figure,
        make_gif=make_gif,
        color=color,
        save_path=save_path,
        fig_name=fig_name,
        gif_name=gif_name,
        save_mesh=save_mesh, save_mesh_path=save_mesh_path,
        save_mesh_name=save_mesh_name
    )
