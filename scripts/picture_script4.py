import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import os

# ---------- 用户可修改 ----------
NPZ_PATH = "formal_test_data.npz"
CATEGORY_NAMES = ["star", "gripper", "T", "V", "O"]

# Set2 调色板挑 5 色（见上一次答复）
set2 = cm.get_cmap("Set2", 8)
BASE_COLORS = np.array([set2(i)[:3] for i in [0, 1, 2, 4, 5]])

V_INTERVALS = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7)]
OUT_DIR = "figs_bulk_shear"
POINT_SIZE_MIN, POINT_SIZE_MAX = 20, 120
# ---------------------------------

# ---- 1. 读数据 ----
raw = np.load(NPZ_PATH, allow_pickle=True)
inputs = raw["inputs"]         # (N, num_elems, k+1)
labels = raw["labels"].astype(float)
num_freq = inputs.shape[2] - 1
assert num_freq == len(CATEGORY_NAMES), "类别数与频率列不匹配"

# ---- 2. 计算 E^{ave}, ν^{ave} (原公式) ----
E_solid = 2.41
nu_solid = 0.35
E_ave_lst, nu_ave_lst = [], []

for c in labels:
    D = np.array([[c[0], c[1], c[2]],
                  [c[1], c[3], c[4]],
                  [c[2], c[4], c[5]]], float)
    S = np.linalg.inv(D)
    S11, S12, S22 = S[0, 0], S[0, 1], S[1, 1]
    E_ave = 0.5 * (1/S11 + 1/S22)
    nu_ave = -0.5 * ((S12/S11) + (S12/S22))
    E_ave_lst.append(E_ave)
    nu_ave_lst.append(nu_ave)

E_ave_arr = np.asarray(E_ave_lst)
nu_ave_arr = np.asarray(nu_ave_lst)

# ---- 3. 计算 μ*, κ* 并归一化 ----
mu_solid = E_solid / (2 * (1 + nu_solid))
kappa_solid = E_solid / (2 * (1 - nu_solid))

mu_arr = E_ave_arr / (2 * (1 + nu_ave_arr))
kappa_arr = E_ave_arr / (2 * (1 - nu_ave_arr))

mu_norm = mu_arr / mu_solid
kappa_norm = kappa_arr / kappa_solid

# ---- 4. 颜色 & 点大小 (主导类别+漂白) ----
COLOR_DICT = dict(zip(CATEGORY_NAMES, BASE_COLORS))
rgb_colors, sizes, v_vals = [], [], []

for grp in inputs:
    freq = grp[:, :-1].mean(axis=0)
    freq /= freq.sum()
    idx_max = np.argmax(freq)
    alpha = freq[idx_max]          # dominance
    color = alpha * BASE_COLORS[idx_max] + (1 - alpha) * np.ones(3)
    rgb_colors.append(color)

    v = float(grp[0, -1])
    v_vals.append(v)
    frac = (v - 0.3) / 0.4
    sizes.append(POINT_SIZE_MIN + frac * (POINT_SIZE_MAX - POINT_SIZE_MIN))

rgb_colors = np.vstack(rgb_colors)
sizes = np.asarray(sizes)
v_vals = np.asarray(v_vals)

# ---- 5. 绘图工具 ----
def add_legend(ax):
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=COLOR_DICT[c], markersize=8,
                      label=c) for c in CATEGORY_NAMES]
    ax.legend(handles=handles, title="Building blocks",
              loc="lower right", frameon=False, fontsize=9)

def plot(mask, title, fname):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(kappa_norm[mask], mu_norm[mask],
               c=rgb_colors[mask], s=sizes[mask],
               edgecolors='k', linewidths=0.15, alpha=0.9)
    add_legend(ax)
    ax.set_xlabel(r"Bulk modulus $\kappa^{(*)}/\kappa_{\mathrm{solid}}$", fontsize=11)
    ax.set_ylabel(r"Shear modulus $\mu^{(*)}/\mu_{\mathrm{solid}}$", fontsize=11)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim(0, 1.05 * kappa_norm.max())
    ax.set_ylim(0, 1.05 * mu_norm.max())
    ax.set_facecolor('white')
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# ---- 6. 输出 ----
os.makedirs(OUT_DIR, exist_ok=True)
# 全数据
plot(np.ones_like(v_vals, bool),
     "Bulk–Shear modulus space (all data)",
     os.path.join(OUT_DIR, "bulk_shear_all.jpg"))

# 分段
for low, high in V_INTERVALS:
    m = (v_vals >= low) & (v_vals < high)
    if not m.any():
        print(f"No samples in v∈[{low:.1f},{high:.1f}) — skipped.")
        continue
    plot(m,
         rf"$v \in [{low:.1f}, {high:.1f})$",
         os.path.join(OUT_DIR, f"bulk_shear_v_{low:.1f}_{high:.1f}.jpg"))

print("Done.")
