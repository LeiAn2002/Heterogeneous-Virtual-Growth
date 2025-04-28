import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# 1. 读取数据库
# -------------------------------------------------
npz_path = "incremental_training_data.npz"        # ← 修改为实际路径
data      = np.load(npz_path, allow_pickle=True)
inputs    = data["inputs"]                        # (N_groups, num_elems, 4)
labels    = data["labels"].astype(float)          # (N_groups, 6)

# -------------------------------------------------
# 2. 计算 E^{ave}/E_solid 与 ν^{ave}
# -------------------------------------------------
E_solid = 2.41
E_norm, nu_vals = [], []

for c in labels:
    D = np.array([[c[0], c[1], c[2]],
                  [c[1], c[3], c[4]],
                  [c[2], c[4], c[5]]], dtype=float)

    S = np.linalg.inv(D)
    S11, S12, S22 = S[0, 0], S[0, 1], S[1, 1]

    E_ave  = 0.5 * (1/S11 + 1/S22)
    nu_ave = -0.5 * ((S12/S11) + (S12/S22))

    E_norm.append(E_ave / E_solid)
    nu_vals.append(nu_ave)

E_norm = np.asarray(E_norm)
nu_vals = np.asarray(nu_vals)

# -------------------------------------------------
# 3. 预先计算颜色 (RGB) 与尺寸 (s)，并记录 v
# -------------------------------------------------
rgb_colors, sizes, v_values = [], [], []

for inp in inputs:
    freq_mean = np.mean(inp[:, :3], axis=0)
    freq_mean /= freq_mean.sum()          # R+G+B = 1
    rgb_colors.append(freq_mean)

    v = float(inp[0, 3])                  # 同一组 v 恒定，取第一行
    v_values.append(v)
    sizes.append(20 + (v - 0.3) / 0.4 * 100)   # v∈[0.3,0.7] → size∈[20,120]

rgb_colors = np.vstack(rgb_colors)
sizes      = np.asarray(sizes)
v_values   = np.asarray(v_values)

# -------------------------------------------------
# 4. 定义 v 区间并逐区间绘图
# -------------------------------------------------
intervals = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7)]

os.makedirs("figs_by_v", exist_ok=True)

for low, high in intervals:
    # 取闭左开右区间：[low, high)
    mask = (v_values >= low) & (v_values < high)

    if not np.any(mask):
        print(f"No data in v ∈ [{low:.1f}, {high:.1f}) — skipped.")
        continue

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(E_norm[mask], nu_vals[mask],
               c=rgb_colors[mask],
               s=sizes[mask],
               edgecolors='k', linewidths=0.15, alpha=0.85)

    ax.set_xlabel(r"$E^{ave}/E_{solid}$", fontsize=12)
    ax.set_ylabel(r"$\nu^{ave}$",        fontsize=12)
    ax.set_title(fr"$v \in [{low:.1f},\,{high:.1f})$", fontsize=14, pad=10)

    ax.set_xlim(0, 1.05 * E_norm.max())
    ax.set_ylim(1.05 * nu_vals.min(), 1.05 * nu_vals.max())

    plt.tight_layout()
    fname = f"figs_by_v/2D_material_space_v_{low:.1f}_{high:.1f}.jpg"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {fname}")

print("Finished plotting by v-intervals.")
