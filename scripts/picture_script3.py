import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import os

# ============================= 用户可配置 =============================
NPZ_PATH      = "test_data.npz"          # 数据库路径
CATEGORY_NAMES = ["H", "T", "V", "TT"]                  # 频率列顺序 (k 列)
PALETTE_NAME  = "tab10"                                 # 色板
V_INTERVALS   = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7)]
OUT_DIR       = "figs_by_v"                             # 输出文件夹
POINT_SIZE_MIN, POINT_SIZE_MAX = 20, 120                # 散点尺寸范围
# 如需自定义具体颜色，可替换下行 4 个 RGB 元组 (0-1 归一化)
BASE_COLORS   = None    # 例如 [(0.89,0.10,0.11), ...]；为 None 时自动用色板
# ====================================================================


# -------------------------------------------------
# 1. 读取数据库
# -------------------------------------------------
data     = np.load(NPZ_PATH, allow_pickle=True)
inputs   = data["inputs"]                        # (N, num_elems, k+1)
labels   = data["labels"].astype(float)          # (N, 6)

num_freq = inputs.shape[2] - 1                  # 检查 k
assert num_freq == len(CATEGORY_NAMES), \
    f"Category count {len(CATEGORY_NAMES)} != frequency columns {num_freq}"

# -------------------------------------------------
# 2. 等效模量 & 泊松比
# -------------------------------------------------
E_solid = 2.41
E_norm, nu_vals = [], []

for c in labels:
    # stiffness matrix (3×3)
    D = np.array([[c[0], c[1], c[2]],
                  [c[1], c[3], c[4]],
                  [c[2], c[4], c[5]]], dtype=float)
    S = np.linalg.inv(D)
    S11, S12, S22 = S[0, 0], S[0, 1], S[1, 1]
    E_ave = 0.5 * (1/S11 + 1/S22)
    nu_ave = -0.5 * ((S12/S11) + (S12/S22))
    E_norm.append(E_ave / E_solid)
    nu_vals.append(nu_ave)

E_norm = np.asarray(E_norm)
nu_vals = np.asarray(nu_vals)

# ========= Step 3. 计算颜色、尺寸、v  (只展示改动部分) =========
# 3-1 预设 4 个基准色（可自行替换）
BASE_COLORS = np.array([
    (31, 119, 180),   # H  → 蓝
    (255, 127, 14),   # T  → 橙
    (44, 160, 44),    # V  → 绿
    (214, 39, 40)     # TT → 红
]) / 255.0           # 归一化到 0-1

COLOR_DICT = dict(zip(CATEGORY_NAMES, BASE_COLORS))  # 供图例使用

rgb_colors, sizes, v_vals = [], [], []

for grp in inputs:
    freq_mean = grp[:, :-1].mean(axis=0)         # (k,)
    freq_mean = freq_mean / freq_mean.sum()

    idx_max      = np.argmax(freq_mean)          # 主导类别
    dominance    = freq_mean[idx_max]            # 其占比 α
    base_color   = BASE_COLORS[idx_max]          # RGB of dominant

    # α=dominance, 1-α 漂白
    color_rgb    = dominance * base_color + (1 - dominance) * np.ones(3)
    rgb_colors.append(color_rgb)

    v = float(grp[0, -1])
    v_vals.append(v)
    frac = (v - 0.3) / 0.4
    sizes.append(POINT_SIZE_MIN + frac * (POINT_SIZE_MAX - POINT_SIZE_MIN))

rgb_colors = np.vstack(rgb_colors)
sizes      = np.asarray(sizes)
v_vals     = np.asarray(v_vals)


# -------------------------------------------------
# 4. 绘图函数
# -------------------------------------------------
def add_legend(ax):
    """Add dummy points for legend showing category–color mapping."""
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=COLOR_DICT[c], markersize=8,
                      label=c) for c in CATEGORY_NAMES]
    ax.legend(handles=handles, title="Building blocks",
              loc="upper right", frameon=False, fontsize=9)

def plot_scatter(mask, title, fname):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(E_norm[mask], nu_vals[mask],
               c=rgb_colors[mask],
               s=sizes[mask],
               edgecolors='k', linewidths=0.15, alpha=0.9)
    add_legend(ax)
    ax.set_xlabel(r"$E^{ave}/E_{solid}$", fontsize=12)
    ax.set_ylabel(r"$\nu^{ave}$",        fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim(0, 1.05*E_norm.max())
    ax.set_ylim(1.05*nu_vals.min(), 1.05*nu_vals.max())
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# -------------------------------------------------
# 5. 输出图形
# -------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# 5-1 全数据
plot_scatter(mask=np.ones_like(v_vals, dtype=bool),
             title="2D material space (all data)",
             fname=os.path.join(OUT_DIR, "material_space_all.jpg"))

# 5-2 分段
for low, high in V_INTERVALS:
    m = (v_vals >= low) & (v_vals < high)
    if not m.any():
        print(f"No samples in v ∈ [{low:.1f}, {high:.1f}) — skipped.")
        continue
    plot_scatter(mask=m,
                 title=fr"$v \in [{low:.1f},\,{high:.1f})$",
                 fname=os.path.join(
                     OUT_DIR, f"material_space_v_{low:.1f}_{high:.1f}.jpg"))

print("Finished all plots.")
