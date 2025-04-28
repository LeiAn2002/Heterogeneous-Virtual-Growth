import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. 读取数据库
# -------------------------------------------------
npz_path = "incremental_training_data.npz"        # <-- 修改为实际路径
data      = np.load(npz_path, allow_pickle=True)
inputs    = data["inputs"]                        # (N, num_elems, 4)
labels    = data["labels"].astype(float)          # (N, 6)

# -------------------------------------------------
# 2. 计算 Eave / Es, νave
# -------------------------------------------------
E_solid = 2.41
E_norm, nu_vals = [], []

for c in labels:
    D = np.array([[c[0], c[1], c[2]],
                  [c[1], c[3], c[4]],
                  [c[2], c[4], c[5]]], dtype=float)

    S    = np.linalg.inv(D)
    S11, S12, S22 = S[0, 0], S[0, 1], S[1, 1]

    E_ave  = 0.5 * (1.0 / S11 + 1.0 / S22)
    nu_ave = -0.5 * ((S12 / S11) + (S12 / S22))

    E_norm.append(E_ave / E_solid)
    nu_vals.append(nu_ave)

E_norm = np.asarray(E_norm)
nu_vals = np.asarray(nu_vals)

# -------------------------------------------------
# 3. 准备颜色 (RGB) 与尺寸 (s)
# -------------------------------------------------
rgb_colors, sizes = [], []

for inp in inputs:
    # 对该组所有单元求平均频率 → (cross, T, O)
    freq_mean = np.mean(inp[:, :3], axis=0)
    freq_mean /= freq_mean.sum()         # 保证 r+g+b = 1
    rgb_colors.append(freq_mean)         # Matplotlib 接受 (R,G,B) 数组

    v = inp[0, 3]                        # 每组 v 都相同，取第一行
    # v∈[0.3,0.7] → size∈[20,120]  (可自行调整)
    sizes.append(20 + (v - 0.3) / 0.4 * 100)

rgb_colors = np.vstack(rgb_colors)
sizes      = np.asarray(sizes)

# -------------------------------------------------
# 4. 画散点图
# -------------------------------------------------
plt.figure(figsize=(6, 5))

plt.scatter(E_norm, nu_vals, c=rgb_colors, s=sizes,
            edgecolors='k', linewidths=0.15, alpha=0.85)

plt.xlabel(r"$E^{ave}/E_{solid}$", fontsize=12)
plt.ylabel(r"$\nu^{ave}$",        fontsize=12)
plt.title("2D material space",    fontsize=14, pad=10)

plt.xlim(0, np.max(E_norm) * 1.05)
plt.ylim(np.min(nu_vals) * 1.05, np.max(nu_vals) * 1.05)
plt.tight_layout()
# plt.show()
plt.savefig("2D_material_space.jpg", dpi=300, bbox_inches="tight")