import numpy as np
import random
import matplotlib.pyplot as plt

from shapely.geometry import LineString, box
from shapely.ops import unary_union
from scipy.interpolate import make_interp_spline


def generate_random_control_points(basic_points, r, outer_count):
    """
    Generate control points for each basic point.
    - The first 'outer_count' points are considered 'outer/pinned' and will NOT be randomized.
    - The remaining points (inner) will be moved randomly within radius r.
    """
    control_points = []
    for i, (bx, by) in enumerate(basic_points):
        if i < outer_count:
            # Keep outer points fixed
            cx, cy = bx, by
        else:
            # Randomize inner points within a circle of radius r
            rr = random.uniform(0, r)
            angle = random.uniform(0, 2*np.pi)
            cx = bx + rr * np.cos(angle)
            cy = by + rr * np.sin(angle)
        control_points.append((cx, cy))
    return control_points


def generate_bspline_curve(points, num_samples=200):
    """
    Create a cubic B-spline curve using scipy's make_interp_spline.
    Note: If the number of points is < 4, we adjust k accordingly to avoid errors.
    Returns arrays (xs, ys) of the discretized curve.
    """
    pts = np.array(points)
    n = len(pts)
    # Parameter: 0,1,2,...,n-1
    t = np.arange(n)
    # Use min(3, n-1) to avoid errors if points are too few
    spline = make_interp_spline(t, pts, k=min(3, n-1))
    t_new = np.linspace(t[0], t[-1], num_samples)
    xy_new = spline(t_new)
    xs, ys = xy_new[:, 0], xy_new[:, 1]
    return xs, ys


def thicken_curve(xs, ys, thickness, cap_style="flat"):
    """
    Thicken a curve (represented by sampled points) into a strip with the specified thickness.
    Uses Shapely's buffer operation for offset.
    The buffer distance is thickness/2 so that the total width becomes 'thickness'.
    """
    line = LineString([(x, y) for x, y in zip(xs, ys)])
    shape_poly = line.buffer(
        thickness / 2,
        resolution=64,
        cap_style=cap_style  # "round", "flat", or "square"
    )
    return shape_poly


def demo_block_generation(
    basic_points,
    outer_count,
    r,
    thickness,
    curve_definitions,
    plot_control_points=True,
    boundary=(-1, -1, 1, 1)
):
    """
    1. Generate random control points from basic_points, 
       but keep the first 'outer_count' points pinned (outer boundary).
    2. Build multiple B-spline curves based on curve_definitions.
    3. Thicken each curve with 'thickness'.
    4. Clip the final shape with a bounding box ([-1,1]×[-1,1]) if desired.
    5. Visualize.
    """
    # 1. Generate control points
    control_points = generate_random_control_points(basic_points, r, outer_count)
    
    # 2. Build and thicken each sub-curve
    shapes = []
    for subset in curve_definitions:
        sub_ctrls = [control_points[i] for i in subset]
        xs, ys = generate_bspline_curve(sub_ctrls, num_samples=300)
        shape_poly = thicken_curve(xs, ys, thickness, cap_style="flat")
        shapes.append(shape_poly)
    
    # Merge all strips
    final_shape = unary_union(shapes)
    
    # 3. Optionally clip by boundary box
    xmin, ymin, xmax, ymax = boundary
    # bounding_box = box(xmin, ymin, xmax, ymax)
    # clipped_shape = final_shape.intersection(bounding_box)
    
    # 4. Plot
    fig, ax = plt.subplots()
    
    # Plot bounding box for reference
    x_box = [xmin, xmin, xmax, xmax, xmin]
    y_box = [ymin, ymax, ymax, ymin, ymin]
    ax.plot(x_box, y_box, color='black', lw=1)
    
    # Plot the clipped shape
    if final_shape.geom_type == 'Polygon':
        x_ext, y_ext = final_shape.exterior.xy
        ax.fill(x_ext, y_ext, color='skyblue', alpha=0.7)
    elif final_shape.geom_type == 'MultiPolygon':
        for poly in final_shape:
            x_ext, y_ext = poly.exterior.xy
            ax.fill(x_ext, y_ext, color='skyblue', alpha=0.7)
    else:
        x_ext, y_ext = final_shape.exterior.xy
        ax.fill(x_ext, y_ext, color='skyblue', alpha=0.7)
    
    if plot_control_points:
        # Also plot the control points (randomized or pinned)
        cpx, cpy = zip(*control_points)
        ax.scatter(cpx, cpy, color='red', marker='o')
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Random Block Generation Demo")
    plt.savefig("random_block.png", dpi=200)


if __name__ == "__main__":
    # --------------------------------------------
    # Define outer points (pinned):
    # Each pair is (exact boundary point, close neighbor).
    # This helps keep the tangent near vertical/horizontal at edges.
    outer_basic_points = [
        (0, -1),   # bottom boundary
        (0, -0.9999999),
        
        (0, 0.9999999),   # top boundary
        (0, 1),    
        
        (-1, 0),   # left boundary
        (-0.9999999, 0),
        
        (0.9999999, 0),   # right boundary
        (1, 0),
    ]
    
    # Define inner points (to be randomized)
    inner_basic_points = [
        (0, -1/3),
        (0, 1/3),
        (-1/3, 0),
        (1/3, 0),
    ]
    
    # Combine them
    basic_points = outer_basic_points + inner_basic_points
    outer_count = len(outer_basic_points)  # first 'outer_count' are pinned
    
    # We want two curves:
    # 1) vertical curve: from bottom boundary to top boundary, including two pinned pairs + 2 inner
    #    => indices [0,1,8,9,2,3]
    # 2) horizontal curve: from left boundary to right boundary, including two pinned pairs + 2 inner
    #    => indices [4,5,10,11,6,7]
    
    curve_definitions = [
        [0, 1, 8, 9, 2, 3],   # bottom->inner->top (vertical)
        [4, 5, 10, 11, 6, 7],   # left->inner->right (horizontal)
    ]
    
    # Random radius for inner points
    r = 0.33
    # Strip thickness
    thickness = 0.2
    
    # Run the demo
    demo_block_generation(
        basic_points=basic_points,
        outer_count=outer_count,
        r=r,
        thickness=thickness,
        curve_definitions=curve_definitions,
        plot_control_points=True
    )
