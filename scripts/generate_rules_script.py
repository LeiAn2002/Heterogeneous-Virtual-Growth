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

from virtual_growth.pair_rules_2d import pair_rules_2d
from virtual_growth.pair_rules_3d import pair_rules_3d
import numpy as np

v_array = np.array([0.2, 0.3, 0.6, 0.9])

dim = 2
match dim:
    case 2:
        block_names = ["corner", "cross", "line", "skew", "t", "v", "x"]
        # pair_rules_2d(block_names, d=0.5, m=0.75, n=0.25, num_elems_d=3, num_elems_m=5,
        #               path_name="virtual_growth_data/2d/")

        pair_rules_2d(block_names, v_array, m=0.75, num_elems_d=3, num_elems_m=5,
                      path_name="virtual_growth_data/2d/")

    case 3:
        block_names = ["corner", "cross_line", "line", "plane_corner",
                       "cross", "plane_cross", "t", "t_line"]
        pair_rules_3d(block_names, d=0.2, m=1.2, n=0.1, num_elems_m=3,
                      path_name="virtual_growth_data/3d/")
