import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as st

from virtual_growth.main import main
from blocks.block_mesh_2d import generate_mesh, load_msh_with_meshio
from homogenization.homogenization_2d import homogenized_elasticity_matrix_2d

def scenario_set_random():
    """
    原有逻辑：每次随机生成 frequency_hints 与 v, 
    对每组重复 runs_per_scenario 次, 并画 C11 的分布.
    如果某一组耗时 > 5 分钟, 则跳过这一组.
    """
    num_scenarios = 20        # 一共做多少组“随机”场景
    runs_per_scenario = 100  # 每组场景重复多少次
    
    # 创建用于显示的图窗布局
    fig, axes = plt.subplots(4, 5, figsize=(12, 12))
    axes = axes.flatten()

    # 基本网格参数(仅作演示,可按需修改)
    mesh_number = 1
    element_number = 6
    mesh_size = (mesh_number, mesh_number)
    element_size = (element_number, element_number)
    m = 0.75

    # 设定文件路径
    save_path   = "designs/2d/"
    design_path = os.path.join(save_path, "symbolic_graph.npy")
    geo_file    = os.path.join(save_path, "mesh.geo")
    msh_file    = os.path.join(save_path, "mesh.msh")

    candidates_list = ["cross", "T", "O"]

    for i in range(num_scenarios):
        freq = np.random.rand(3)
        freq /= freq.sum()  # 归一化
        v = np.random.uniform(0.3, 0.7)

        c11_list = []
        
        for run_idx in range(runs_per_scenario):
            # 如果运行时间 > 5 分钟, 跳过此组

            try:
                num_elems = mesh_number * mesh_number
                # 将 freq 扩展到每个单元
                frequency_hints = np.tile(freq, (num_elems, 1))

                v_array = np.random.uniform(low=v, high=v, size=(num_elems, 2))
                # 这里 r 随机固定(比如全是 0.1)
                r_array = np.random.uniform(low=0.1, high=0.1, size=(num_elems,))

                # 虚拟生长
                main(
                    mesh_size, element_size, 
                    candidates_list,
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

                # 生成网格
                generate_mesh(design_path, geo_file, msh_file)

                # 做同质化计算
                nodes, tri_elems, quad_elems = load_msh_with_meshio(msh_file)
                mat_table = {
                    "E": 30,
                    "nu": 0.25,
                    "PSflag": "PlaneStress",
                    "RegMesh": False,
                    "thickness": 1.0,
                }
                K_eps = homogenized_elasticity_matrix_2d(
                    nodes, tri_elems, quad_elems, mat_table
                )

                c11_list.append(K_eps[0, 0])
            except Exception as e:
                print(f"Error in run {run_idx} of scenario {i}: {e}")
                continue

        # 绘图
        ax = axes[i]
        if len(c11_list) == 0:
            # 如果因为超时或出错导致该组无数据，就直接跳过
            ax.set_title(f"Scenario {i} skipped/no data")
            continue

        # 直方图
        n, bins, patches = ax.hist(c11_list, bins=10, color="green", alpha=0.6)
        ax.set_xlabel(r"$D_{11}$")
        ax.set_ylabel("Frequency", color="green")

        # 拟合+概率密度曲线
        ax2 = ax.twinx()
        mu, sigma = st.norm.fit(c11_list)
        x = np.linspace(min(c11_list), max(c11_list), 100)
        pdf = st.norm.pdf(x, mu, sigma)
        ax2.plot(x, pdf, "b-", linewidth=2)
        ax2.set_ylabel("Probability", color="blue")

        # 饼图
        inset_ax = inset_axes(ax, width=0.5, height=0.5, loc='upper right')
        pie_colors = ["#f64e4e", "#4ef56d", "#4e90f6"]
        wedges, texts = inset_ax.pie(freq, labels=None, colors=pie_colors,
                                     startangle=90)
        circle = plt.Circle((0, 0), 0.5, color='white')
        inset_ax.add_artist(circle)
        inset_ax.set_aspect("equal")

        # v 标注
        ax.text(
            0.95, 0.7,
            f"v={v:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            ha='right', 
            va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        ax.set_title(f"Scenario {i}")

    fig.tight_layout()
    fig.savefig("random_params_histograms.png", dpi=300)
    plt.close(fig)
    print("Scenario set: random_params finished. Saved to random_params_histograms.png")


def scenario_set_A():
    """
    算例 A: 固定 frequency_hints, v 从 0.2 到 0.8 递增.
    其余逻辑与随机场景类似, 每个 (v) 都跑 runs_per_scenario 次, 画直方图.
    若单组耗时 > 5 分钟则跳过.
    """
    # 定义一组固定 freq, 在这演示 (比如 0.5, 0.3, 0.2)
    freq = np.array([0.5, 0.3, 0.2])
    freq /= freq.sum()

    vs = np.arange(0.3, 0.71, 0.05)  # v 从 0.2 到 0.8, 步长 0.1
    runs_per_scenario = 100        # 每组重复次数(可按需调整)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    mesh_number = 1
    element_number = 6
    mesh_size = (mesh_number, mesh_number)
    element_size = (element_number, element_number)
    m = 0.75

    save_path   = "designs/2d/"
    design_path = os.path.join(save_path, "symbolic_graph.npy")
    geo_file    = os.path.join(save_path, "mesh.geo")
    msh_file    = os.path.join(save_path, "mesh.msh")

    candidates_list = ["cross", "T", "O"]

    for i, v in enumerate(vs):
        c11_list = []

        for run_idx in range(runs_per_scenario):
            try:
                num_elems = mesh_number * mesh_number
                frequency_hints = np.tile(freq, (num_elems, 1))
                v_array = np.random.uniform(low=v, high=v, size=(num_elems, 2))
                # 固定 r=0.1(或随机都可)
                r_array = np.random.uniform(low=0.1, high=0.1, size=(num_elems,))

                main(
                    mesh_size, element_size, 
                    candidates_list,
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
                    "E": 30,
                    "nu": 0.25,
                    "PSflag": "PlaneStress",
                    "RegMesh": False,
                    "thickness": 1.0,
                }
                K_eps = homogenized_elasticity_matrix_2d(
                    nodes, tri_elems, quad_elems, mat_table
                )
                c11_list.append(K_eps[0, 0])
            except Exception as e:
                print(f"Error in run {run_idx} for v={v:.2f}: {e}")
                continue

        if i < len(axes):
            ax = axes[i]
        else:
            # 万一子图不够，直接跳过
            continue

        if len(c11_list) == 0:
            ax.set_title(f"v={v:.2f} skipped/no data")
            continue

        # 绘图
        n, bins, patches = ax.hist(c11_list, bins=10, color="green", alpha=0.6)
        ax.set_xlabel(r"$D_{11}$")
        ax.set_ylabel("Freq.", color="green")

        ax2 = ax.twinx()
        mu, sigma = st.norm.fit(c11_list)
        x = np.linspace(min(c11_list), max(c11_list), 100)
        pdf = st.norm.pdf(x, mu, sigma)
        ax2.plot(x, pdf, "b-", linewidth=2)
        ax2.set_ylabel("Prob.", color="blue")

        inset_ax = inset_axes(ax, width=0.4, height=0.4, loc='upper right')
        pie_colors = ["#f64e4e", "#4ef56d", "#4e90f6"]
        wedges, texts = inset_ax.pie(freq, labels=None, colors=pie_colors, startangle=90)
        circle = plt.Circle((0, 0), 0.5, color='white')
        inset_ax.add_artist(circle)
        inset_ax.set_aspect("equal")

        ax.text(
            0.95, 0.7,
            f"v={v:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            ha='right', 
            va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        ax.set_title(f"Scenario A: v={v:.2f}")

    fig.tight_layout()
    fig.savefig("scenario_set_A.png", dpi=300)
    plt.close(fig)
    print("Scenario set A finished. Saved to scenario_set_A.png")


def scenario_set_B():
    """
    算例 B: 固定 frequency_hints 与 v, 令 r 从 0.1 到 0.6 递增.
    其余逻辑类似, 每组场景跑 runs_per_scenario 次, 超时则跳过.
    """
    # 固定 freq
    freq = np.array([0.5, 0.3, 0.2])
    freq /= freq.sum()

    # 固定 v
    v_fixed = 0.3

    # 令 r 从 0.1 到 0.6(步长 0.1)
    rs = np.arange(0.1, 0.7, 0.05)
    runs_per_scenario = 100

    fig, axes = plt.subplots(3, 4, figsize=(10, 6))
    axes = axes.flatten()

    mesh_number = 1
    element_number = 6
    mesh_size = (mesh_number, mesh_number)
    element_size = (element_number, element_number)
    m = 0.75

    save_path   = "designs/2d/"
    design_path = os.path.join(save_path, "symbolic_graph.npy")
    geo_file    = os.path.join(save_path, "mesh.geo")
    msh_file    = os.path.join(save_path, "mesh.msh")

    candidates_list = ["cross", "T", "L"]

    for i, r_val in enumerate(rs):
        c11_list = []
        
        for run_idx in range(runs_per_scenario):

            try:
                num_elems = mesh_number * mesh_number
                frequency_hints = np.tile(freq, (num_elems, 1))

                # v_array: 这里固定 v = 0.5
                v_array = np.random.uniform(low=v_fixed, high=v_fixed, size=(num_elems, 2))
                # r_array: 由 r_val 决定
                r_array = np.random.uniform(low=r_val, high=r_val, size=(num_elems,))

                main(
                    mesh_size, element_size, 
                    candidates_list,
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
                    "E": 30,
                    "nu": 0.25,
                    "PSflag": "PlaneStress",
                    "RegMesh": False,
                    "thickness": 1.0,
                }
                K_eps = homogenized_elasticity_matrix_2d(
                    nodes, tri_elems, quad_elems, mat_table
                )
                c11_list.append(K_eps[0, 0])
            except Exception as e:
                print(f"Error in run {run_idx} for r={r_val:.2f}: {e}")
                continue

        if i < len(axes):
            ax = axes[i]
        else:
            continue

        if len(c11_list) == 0:
            ax.set_title(f"r={r_val:.2f} skipped/no data")
            continue

        n, bins, patches = ax.hist(c11_list, bins=10, color="green", alpha=0.6)
        ax.set_xlabel(r"$D_{11}$")
        ax.set_ylabel("Freq.", color="green")

        ax2 = ax.twinx()
        mu, sigma = st.norm.fit(c11_list)
        x = np.linspace(min(c11_list), max(c11_list), 100)
        pdf = st.norm.pdf(x, mu, sigma)
        ax2.plot(x, pdf, "b-", linewidth=2)
        ax2.set_ylabel("Prob.", color="blue")

        inset_ax = inset_axes(ax, width=0.4, height=0.4, loc='upper right')
        pie_colors = ["#f64e4e", "#4ef56d", "#4e90f6"]
        wedges, texts = inset_ax.pie(freq, labels=None, colors=pie_colors, startangle=90)
        circle = plt.Circle((0, 0), 0.5, color='white')
        inset_ax.add_artist(circle)
        inset_ax.set_aspect("equal")

        # 标注一下 v 与 r
        ax.text(
            0.95, 0.7,
            f"v={v_fixed:.2f}\nr={r_val:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            ha='right', 
            va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        ax.set_title(f"Scenario B: r={r_val:.2f}")

    fig.tight_layout()
    fig.savefig("scenario_set_B.png", dpi=300)
    plt.close(fig)
    print("Scenario set B finished. Saved to scenario_set_B.png")


def main_function():
    """
    主函数: 依次调用三套场景.
    """
    # 1) 原先的随机多组参数
    # scenario_set_random()
    
    # # 2) 算例 A: 固定 freq, 让 v 从 0.2 到 0.8 递增
    # scenario_set_A()
    
    # 3) 算例 B: 固定 freq 与 v, 让 r 从 0.1 到 0.6 递增
    scenario_set_B()


if __name__ == "__main__":
    main_function()
