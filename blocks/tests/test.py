import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString
from shapely.ops import unary_union
from scipy.interpolate import make_interp_spline


def generate_bspline_curve(points, num_samples=200):
    """
    Create a cubic B-spline curve using scipy's make_interp_spline.
    If points are fewer than 4, it auto-adjusts k to avoid errors.
    Returns arrays (xs, ys) of the discretized curve.
    """
    pts = np.array(points)
    n = len(pts)
    # Parameter: 0,1,2,...,n-1
    t = np.arange(n)
    # Use min(3, n-1) to avoid errors if fewer than 4 points
    spline = make_interp_spline(t, pts, k=min(3, n-1))
    t_new = np.linspace(t[0], t[-1], num_samples)
    xy_new = spline(t_new)
    xs, ys = xy_new[:, 0], xy_new[:, 1]
    return xs, ys


def variable_thickness_spline(points, thickness_func, num_samples=300, cap_style="round"):
    """
    Build a 'variable thickness' strip around a B-spline curve.
    
    Steps:
      1) Discretize the spline into (xs, ys).
      2) Split into many small segments: [(p0,p1), (p1,p2), ...].
      3) For each segment, compute a thickness (using thickness_func).
      4) buffer() each small segment and union all polygons.
      
    Args:
        points: control points for the B-spline [(x0, y0), (x1, y1), ...].
        thickness_func: a function f(i, total_segments) -> thickness,
                        or f(t) with t in [0,1] if we do param-based.
        num_samples: number of sample points for the B-spline.
        cap_style: "round", "square", or "flat".
        
    Returns:
        A shapely Polygon or MultiPolygon representing the final variable-width strip.
    """
    xs, ys = generate_bspline_curve(points, num_samples=num_samples)
    n = len(xs)
    
    # We'll store each segment's buffer in a list
    segment_polys = []
    
    # total_segments = n - 1
    for i in range(n - 1):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i+1], ys[i+1]
        
        # We can define t in [0,1] for parameter along the curve
        # e.g. t = i / (n - 2)
        t = i / (n - 2) if (n - 2) > 0 else 0
        
        # thickness for this small segment
        T_i = thickness_func(t)
        
        # Build a small linestring for the segment
        seg_line = LineString([(x1, y1), (x2, y2)])
        
        # buffer it by half the thickness
        seg_poly = seg_line.buffer(T_i / 2, resolution=32, cap_style=cap_style)
        segment_polys.append(seg_poly)
    
    # Union all segment polygons
    final_strip = unary_union(segment_polys)
    return final_strip


# ------------------ DEMO USAGE ------------------
if __name__ == "__main__":
    # Example 1: A simple set of control points forming a gentle curve
    control_points = [
        (0, 0),
        (1, 1),
        (2, 0),
        (3, 1),
        (4, 0)
    ]
    
    # We'll define a thickness function that linearly increases
    # from 0.2 at the start to 1.0 at the end of the curve.
    def thickness_func_linear(t):
        return 0.2 + 0.8 * t
    
    # Build the variable-thickness strip
    shape_var = variable_thickness_spline(
        points=control_points,
        thickness_func=thickness_func_linear,
        num_samples=200,
        cap_style="round"
    )
    
    # For plotting, let's also show the center line
    xs_center, ys_center = generate_bspline_curve(control_points, num_samples=200)
    
    # --- Visualization ---
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    # Fill the variable-width shape
    if shape_var.geom_type == 'Polygon':
        x_ext, y_ext = shape_var.exterior.xy
        ax.fill(x_ext, y_ext, color='skyblue', alpha=0.7, label='Variable-Width Strip')
    elif shape_var.geom_type == 'MultiPolygon':
        for poly in shape_var:
            x_ext, y_ext = poly.exterior.xy
            ax.fill(x_ext, y_ext, color='skyblue', alpha=0.7)
    
    # Plot the center line
    ax.plot(xs_center, ys_center, 'k--', label='Center Spline')
    # Plot the original control points
    cpx = [p[0] for p in control_points]
    cpy = [p[1] for p in control_points]
    ax.scatter(cpx, cpy, color='red', marker='o', zorder=5, label='Control Points')
    
    ax.set_aspect('equal', 'box')
    ax.set_title("Variable Thickness Spline Demo")
    ax.legend()
    plt.savefig("test.png", dpi=200)
