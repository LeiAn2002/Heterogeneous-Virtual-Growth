import numpy as np
import random
import matplotlib.pyplot as plt

from shapely.geometry import LineString, box, Point
from shapely.ops import unary_union
from scipy.interpolate import make_interp_spline
from PIL import Image


def image_to_mask_array(png_file, threshold=10):
    """
    Read a saved PNG image and classify the "blue parts" as 1, otherwise as 0.
    Here, a simple rule is used: any pixel that is not pure white (255, 255, 255) is considered as 1.

    Alternatively, you can apply a more refined check on (r, g, b), such as determining if the blue component is significantly greater than the red and green components.

    Parameters:
        threshold: Tolerance level for color classification.
    """
    img = Image.open(png_file).convert('RGB') 
    arr = np.array(img) 
    H, W, _ = arr.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            r, g, b = arr[y, x]
            if not (abs(r-255) < threshold and abs(g-255) < threshold and abs(b-255) < threshold):
                mask[y, x] = 1
    return mask


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    segment_vec = np.array([x2 - x1, y2 - y1])
    point_vec = np.array([px - x1, py - y1])

    segment_length_sq = np.dot(segment_vec, segment_vec)
    if segment_length_sq == 0:
        return np.linalg.norm(point_vec)
    t = np.dot(point_vec, segment_vec) / segment_length_sq
    t = max(0, min(1, t))
    nearest = np.array([x1, y1]) + t * segment_vec
    return np.linalg.norm(np.array([px, py]) - nearest)


def generate_random_control_points(basic_points, r, inner_points_number):
    """
    Generate control points for each basic point.
    - The first 'outer_count' points are considered 'outer/pinned' and will NOT be randomized.
    - The remaining points (inner) will be moved randomly within radius r.
    """
    control_points = basic_points.copy()
    for i in range(inner_points_number):
        rr = random.uniform(0, r)
        angle = random.uniform(0, 2*np.pi)
        cx = rr * np.cos(angle)
        cy = rr * np.sin(angle)
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
    # t_new = np.linspace(t[0], t[-1], num_samples)
    # xy_new = spline(t_new)
    # xs, ys = xy_new[:, 0], xy_new[:, 1]
    # return xs, ys
    xs = []
    ys = []
    for i in range(n - 1):
        t_sub = np.linspace(i, i + 1, num_samples)
        xy_sub = spline(t_sub)
        x_sub = xy_sub[:, 0]
        y_sub = xy_sub[:, 1]
        xs.append(x_sub)
        ys.append(y_sub)
    return xs, ys


def get_original_thickness(xs, ys, forbidden_edges):
    """
    Compute the thickness of a strip at each point.
    The thickness is the minimum distance to any forbidden edge.
    """
    thicknesses = []
    for x, y in zip(xs, ys):
        min_dist = min(
            point_to_segment_distance(x, y, *edge)
            for edge in forbidden_edges
        )
        thicknesses.append(min_dist)
    return thicknesses


def modify_thickness_with_vf(thickness, vf):
    N = len(thickness)
    transition = np.linspace(vf[0], vf[1], N)
    modified_thickness = thickness * transition
    return modified_thickness


def thicken_curve(xs, ys, forbidden_edges, vf, cap_style="flat"):
    """
    Thicken a curve (represented by sampled points) into a strip with the specified thickness.
    Uses Shapely's buffer operation for offset.
    The buffer distance is thickness/2 so that the total width becomes 'thickness'.
    """
    n = len(xs)
    ori_thickness = get_original_thickness(xs, ys, forbidden_edges)
    thickness = modify_thickness_with_vf(ori_thickness, vf)
    segment_polys = []
    for i in range(n-1):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i+1], ys[i+1]
        t = thickness[i]
        seg_line = LineString([(x1, y1), (x2, y2)])
        seg_poly = seg_line.buffer(t, resolution=32, cap_style=cap_style)
        segment_polys.append(seg_poly)

    final_strip = unary_union(segment_polys)
    return final_strip


def demo_block_generation(
    outer_points,
    inner_points_number,
    r,
    curve_definitions,
    vf,
    forbidden_edges_set,
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
    control_points = generate_random_control_points(outer_points, r, inner_points_number)
    
    # 2. Build and thicken each sub-curve
    shapes = []
    count = 0
    x_corr = []
    y_corr = []
    for subset in curve_definitions:
        sub_ctrls = [control_points[i] for i in subset]
        forbidden_edges = forbidden_edges_set[count]
        v = vf[count]
        xs, ys = generate_bspline_curve(sub_ctrls, num_samples=500)
        for i in range(len(xs)):
            x_sub = xs[i]
            y_sub = ys[i]
            v_sub = [v[i], v[i+1]]
            shape_poly = thicken_curve(x_sub, y_sub, forbidden_edges, v_sub, cap_style="flat")
            shapes.append(shape_poly)
            x_corr.append(x_sub)
            y_corr.append(y_sub)
        count += 1
    
    # Merge all strips
    final_shape = unary_union(shapes)
    
    # 3. Optionally clip by boundary box
    xmin, ymin, xmax, ymax = boundary
    # bounding_box = box(xmin, ymin, xmax, ymax)
    # clipped_shape = final_shape.intersection(bounding_box)
    
    # 4. Plot
    fig, ax = plt.subplots()
    ax.axis("off")
    # Plot bounding box for reference
    # x_box = [xmin, xmin, xmax, xmax, xmin]
    # y_box = [ymin, ymax, ymax, ymin, ymin]
    # ax.plot(x_box, y_box, color='black', lw=1)
    
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
    
    # if plot_control_points:
    #     # Also plot the control points (randomized or pinned)
    #     cpx, cpy = zip(*control_points)
    #     ax.scatter(cpx, cpy, color='red', marker='o')
    
    # for xs, ys in zip(x_corr, y_corr):
    #     ax.plot(xs, ys, 'k-', alpha=0.5)  # plot the curves
    # ax.set_aspect('equal', 'box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.set_title("Random Block Generation Demo")
    out_image = "random_block_idea_2.png"
    plt.savefig(out_image, dpi=20)
    
    mask = image_to_mask_array(out_image)
    plt.figure()
    plt.imshow(mask, origin='upper', cmap='gray')
    plt.title("Raster Mask from PNG")
    plt.colorbar()
    plt.savefig("pixel.png")


def cross_block(radius, vf):
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
    inner_points_number = 4
    
    curve_definitions = [
        [0, 1, 8, 9, 2, 3],   # bottom->inner->top (vertical)
        [4, 5, 10, 11, 6, 7],   # left->inner->right (horizontal)
    ]
    
    # Random radius for inner points
    r = radius

    forbidden_edges = [[(-1, -1, -1, 1),
                        (1, -1, 1, 1),
                        (-1, -1, -vf[0][0], -1),
                        (vf[0][0], -1, 1, -1), 
                        (-1, 1, -vf[0][-1], 1), 
                        (vf[0][-1], 1, 1, 1)], 
                       [(-1, -1, 1, -1),
                        (-1, 1, 1, 1),
                        (-1, -1, -1, -vf[1][0]),
                        (-1, vf[1][0], -1, 1), 
                        (1, -1, 1, -vf[1][-1]), 
                        (1, vf[1][-1], 1, 1)
                        ]]   # 为什么有vf项：确保相交的部分也只与边界指定位置相交
    return outer_basic_points, inner_points_number, curve_definitions, r, forbidden_edges


def L_block(radius, vf):
    outer_basic_points = [        
        (-1/3, 1),   # left boundary
        (-1/3, 0.99999999),
        
        (0.9999999, 0),   # right boundary
        (1, 0),
    ]

    inner_points_number = 2

    curve_definitions = [[0, 1, 4, 5, 2, 3]]

    r = radius

    forbidden_edges = [[(-1, -1, -1, 1),
                        (-1, -1, 1, -1),
                        (1/3, 1, 1, 1),
                        (-1, 1, -1+2/3*(1-vf[0][0]), 1),
                        (-1/3+2/3*(1-vf[0][0]), 1, 1/3, 1),
                        (1, -1, 1, -vf[0][-1]),
                        (1, vf[0][-1], 1, 1)]]

    return outer_basic_points, inner_points_number, curve_definitions, r, forbidden_edges


if __name__ == "__main__":

    radius = 0.7
    vf = [[0.4, 0.4, 0.1, 0.5, 0.4, 0.4], [0.3, 0.3, 0.3, 0.2, 0.4, 0.4]]
    # vf = [[0.3, 0.3, 0.3, 0.2, 0.4, 0.4]]
    outer_basic_points, inner_points_number, curve_definitions, r, forbidden_edges = cross_block(radius, vf)
    # Run the demo
    demo_block_generation(
        outer_points=outer_basic_points,
        inner_points_number=inner_points_number,
        r=r,
        curve_definitions=curve_definitions,
        vf=vf,
        forbidden_edges_set=forbidden_edges,
        plot_control_points=True
    )
