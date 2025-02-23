import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf


def generate_interior_points(n_points=100, seed=0):

    np.random.seed(seed)
    xs = np.random.rand(n_points)
    ys = np.random.rand(n_points)

    def random_field(x, y):
        return (
            0.5*np.exp(-((x-0.3)**2+(y-0.3)**2)/0.01) + 
            0.3*np.exp(-((x-0.7)**2+(y-0.8)**2)/0.02) +
            0.2*np.exp(-((x-0.5)**2+(y-0.5)**2)/0.05) +
            0.1 * np.random.randn(*x.shape)
        )

    values = random_field(xs, ys)
    return xs, ys, values


def generate_boundary_points(n_per_edge=2, phi_value=0.0):

    x_b1 = np.linspace(0.4,0.6,n_per_edge)
    y_b1 = np.zeros_like(x_b1)

    x_b2 = np.linspace(0.4,0.6,n_per_edge)
    y_b2 = np.ones_like(x_b2)

    y_b3 = np.linspace(0.4,0.6,n_per_edge)
    x_b3 = np.zeros_like(y_b3)

    y_b4 = np.linspace(0.4,0.6,n_per_edge)
    x_b4 = np.ones_like(y_b4)

    xs = np.concatenate([x_b1, x_b2, x_b3, x_b4])
    ys = np.concatenate([y_b1, y_b2, y_b3, y_b4])
    vals = np.ones_like(xs) * phi_value

    return xs, ys, vals


def rbf_interpolation(xi, yi, zi, function='multiquadric', epsilon=0.1):

    rbf_func = Rbf(xi, yi, zi, function=function, epsilon=epsilon)
    return lambda x, y: rbf_func(x, y)


x_in, y_in, z_in = generate_interior_points(n_points=80, seed=42)

x_bnd1, y_bnd1, z_bnd1 = generate_boundary_points(n_per_edge=20, phi_value=0.2)

x_train1 = np.concatenate([x_in, x_bnd1])
y_train1 = np.concatenate([y_in, y_bnd1])
z_train1 = np.concatenate([z_in, z_bnd1])

rbf_func1 = rbf_interpolation(x_train1, y_train1, z_train1)

x_bnd2, y_bnd2, z_bnd2 = generate_boundary_points(n_per_edge=20, phi_value=-0.1)

x_train2 = np.concatenate([x_in, x_bnd2])
y_train2 = np.concatenate([y_in, y_bnd2])
z_train2 = np.concatenate([z_in, z_bnd2])

rbf_func2 = rbf_interpolation(x_train2, y_train2, z_train2)

Ngrid = 200
grid_x = np.linspace(0,1,Ngrid)
grid_y = np.linspace(0,1,Ngrid)
GX, GY = np.meshgrid(grid_x, grid_y)

phi1 = rbf_func1(GX, GY)
phi2 = rbf_func2(GX, GY)

fig, axes = plt.subplots(1, 2, figsize=(10,5))

cs1 = axes[0].contourf(GX, GY, phi1, levels=50, cmap='RdYlBu')
c0_1 = axes[0].contour(GX, GY, phi1, levels=[0.0], colors='black', linewidths=2)
axes[0].clabel(c0_1, inline=True, fmt="phi=0")
axes[0].set_title("scene1:boundary phi=+0.2")
axes[0].set_aspect('equal', 'box')


cs2 = axes[1].contourf(GX, GY, phi2, levels=50, cmap='RdYlBu')
c0_2 = axes[1].contour(GX, GY, phi2, levels=[0.0], colors='black', linewidths=2)
axes[1].clabel(c0_2, inline=True, fmt="phi=0")
axes[1].set_title("scene2:boundary phi=-0.1")
axes[1].set_aspect('equal', 'box')

plt.tight_layout()
plt.savefig("rbf_interpolation.png")
