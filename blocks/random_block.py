import numpy as np
import random
import matplotlib.pyplot as plt
import time
import io

from shapely.geometry import LineString
from shapely.ops import unary_union
from scipy.interpolate import make_interp_spline
from scipy.sparse import coo_matrix
from scipy.ndimage import binary_fill_holes
from PIL import Image


def build_filter_matrix(H, W, r):
    """
    Build a sparse matrix for a 2D 'linear filter' of radius r on a (H x W) grid.
    Each row e corresponds to one pixel (y, x).
    Each column i also corresponds to one pixel.
    The entry (e, i) = max(0, r - dist(e,i)) if dist(e,i) < r, else 0.

    Returns:
        H_sp: sparse matrix of shape [N, N], N = H*W
        row_sum: an array of row sums (for normalizing).
    """
    # We'll flatten pixel index as e = y*W + x
    # Helper function: 2D->1D
    def idx(x, y):
        return y*W + x

    row_idx = []
    col_idx = []
    data = []

    r_int = int(np.ceil(r))
    for y in range(H):
        for x in range(W):
            e = idx(x, y)

            # local window in [x-r_int, x+r_int] etc
            x_min = max(0, x - r_int)
            x_max = min(W-1, x + r_int)
            y_min = max(0, y - r_int)
            y_max = min(H-1, y + r_int)

            for yy in range(y_min, y_max+1):
                for xx in range(x_min, x_max+1):
                    dist = np.sqrt((xx - x)**2 + (yy - y)**2)
                    if dist < r:
                        w = r - dist
                        col = idx(xx, yy)
                        row_idx.append(e)
                        col_idx.append(col)
                        data.append(w)

    N = H * W
    H_coo = coo_matrix((data, (row_idx, col_idx)), shape=(N, N), dtype=np.float32)
    H_sp = H_coo.tocsr()
    row_sum = np.array(H_sp.sum(axis=1)).ravel()  # shape=(N,)
    return H_sp, row_sum


def linear_filter(mask, r, H_sp=None, row_sum=None):
    """
    Apply a radius-r linear filter to the mask using the sparse matrix approach.
    If H_sp and row_sum are not given, we'll build them on the fly (slower).

    mask: 2D numpy array (H x W)
    r:    filter radius (float)
    H_sp, row_sum: optional precomputed filter matrix and row sums.

    Returns:
        new_mask: 2D array of same shape, filtered by the linear kernel.
    """
    H, W = mask.shape
    N = H*W
    # flatten mask
    x_vec = mask.reshape(-1).astype(np.float32)

    # Build filter matrix if not provided
    if H_sp is None or row_sum is None:
        H_sp, row_sum = build_filter_matrix(H, W, r)

    # multiply
    x_filtered = H_sp.dot(x_vec)
    # row-wise normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        x_filtered /= row_sum

    # reshape
    new_mask = x_filtered.reshape(H, W)
    return new_mask


def heaviside(mask, beta=32):
    """
    Apply a Heaviside function to the mask.
    The Heaviside function is approximated by a sigmoid with slope beta.
    """
    return (np.tanh(beta * 0.5) + np.tanh(beta * (mask - 0.5))) / (2 * np.tanh(beta * 0.5))


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
    """
    Compute the nearest distance between a point and a line segment.
    The line segment is defined by two points (x1, y1) and (x2, y2).
    """
    segment_vec = np.array([x2 - x1, y2 - y1])
    point_vec = np.array([px - x1, py - y1])

    segment_length_sq = np.dot(segment_vec, segment_vec)
    if segment_length_sq == 0:
        return np.linalg.norm(point_vec)
    t = np.dot(point_vec, segment_vec) / segment_length_sq
    t = max(0, min(1, t))
    nearest = np.array([x1, y1]) + t * segment_vec
    return np.linalg.norm(np.array([px, py]) - nearest)


def generate_random_control_points(basic_points, r, outer_count):
    """
    Generate control points for each basic point.
    - The first 'outer_count' points are considered 'outer/pinned' and will NOT be randomized.
    - The remaining points (inner) will be moved randomly within radius r.
    """
    # random.seed(42)
    control_points = []
    for i, (bx, by) in enumerate(basic_points):
        if i < outer_count:
            # Keep outer points fixed
            cx, cy = bx, by
        else:
            # Randomize inner points within a circle of radius r
            while True:
                rr = random.uniform(0, r)
                angle = random.uniform(0, 2 * np.pi)
                cx = bx + rr * np.cos(angle)
                cy = by + rr * np.sin(angle)
                if abs(cx) + abs(cy) < 1.0:
                    break
        control_points.append((cx, cy))
    return control_points


def generate_bspline_curve(points, num_samples=500):
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
    """
    Modify the thickness values using a variable factor (vf) array.
    The thickness at each point is multiplied by the corresponding factor.
    """
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


def block_generation(
    basic_points,
    outer_count,
    r,
    curve_definitions,
    vf,
    forbidden_edges_set,
    r_filter,
    boundary=[-1, -1, 1, 1]
):
    """
    1. Generate random control points from basic_points, 
       but keep the first 'outer_count' points pinned (outer boundary).
    2. Build multiple B-spline curves based on curve_definitions.
    3. Thicken each curve with 'thickness'.
    """
    # 1. Generate control points
    control_points = generate_random_control_points(basic_points, r, outer_count)
    
    # 2. Build and thicken each sub-curve
    shapes = []
    count = 0
    x_corr = []
    y_corr = []
    for subset in curve_definitions:
        sub_ctrls = [control_points[i] for i in subset]
        forbidden_edges = forbidden_edges_set[count]
        v = vf[count]
        xs, ys = generate_bspline_curve(sub_ctrls, num_samples=100)
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

    # cpx, cpy = zip(*control_points)
    # ax.scatter(cpx, cpy, color='red', marker='o')

    # for xs, ys in zip(x_corr, y_corr):
    #     ax.plot(xs, ys, 'k-', alpha=0.5)  # plot the curves

    ax.set_aspect('equal', 'box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=15)
    # plt.savefig("Origin.png", bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

    mask = image_to_mask_array(buf)

    mask = linear_filter(mask, r_filter)

    mask = heaviside(mask, beta=256)

    mask = binary_fill_holes(mask)

    return mask

#     plt.figure()
#     plt.imshow(mask, origin='upper', cmap='gray_r')
#     plt.title("Random Block")
#     plt.colorbar()
#     plt.savefig("Random Block.png")


# def cross_block(radius, vf, boundary_vf):
#     # Define outer points (pinned):
#     # Each pair is (exact boundary point, close neighbor).
#     # This helps keep the tangent near vertical/horizontal at edges.
#     outer_basic_points = [
#         (0, -1),   # bottom boundary
#         (0, -0.9),

#         (0, 0.9),   # top boundary
#         (0, 1),

#         (-1, 0),   # left boundary
#         (-0.9, 0),

#         (0.9, 0),   # right boundary
#         (1, 0),
#     ]

#     # Define inner points (to be randomized)
#     inner_basic_points = [
#         (0, -1/3),
#         (0, 1/3),
#         (-1/3, 0),
#         (1/3, 0),
#     ]

#     basic_points = outer_basic_points + inner_basic_points
#     outer_count = len(outer_basic_points)

#     curve_definitions = [
#         [0, 1, 8, 9, 2, 3],   # bottom->inner->top (vertical)
#         [4, 5, 10, 11, 6, 7],   # left->inner->right (horizontal)
#     ]

#     # Random radius for inner points
#     r = radius

#     forbidden_edges = [[(-1, -1, -1, 1),
#                         (1, -1, 1, 1),
#                         (-1, -1, -boundary_vf[0][0], -1),
#                         (boundary_vf[0][0], -1, 1, -1),
#                         (-1, 1, -boundary_vf[0][-1], 1),
#                         (boundary_vf[0][-1], 1, 1, 1)],
#                        [(-1, -1, 1, -1),
#                         (-1, 1, 1, 1),
#                         (-1, -1, -1, -boundary_vf[1][0]),
#                         (-1, boundary_vf[1][0], -1, 1),
#                         (1, -1, 1, -boundary_vf[1][-1]),
#                         (1, boundary_vf[1][-1], 1, 1)
#                         ]]

#     return basic_points, outer_count, curve_definitions, r, forbidden_edges


# def L_block(radius, vf, boundary_vf):
#     outer_basic_points = [        
#         (-1/3, 1),   # left boundary
#         (-1/3, 0.99999999),
        
#         (0.9999999, 0),   # right boundary
#         (1, 0),
#     ]

#     inner_basic_points = [
#         (-1/3, 0.5),
#         (0.5, 0)
#     ]

#     basic_points = outer_basic_points + inner_basic_points
#     outer_count = len(outer_basic_points)

#     curve_definitions = [[0, 1, 4, 5, 2, 3]]

#     r = radius

#     forbidden_edges = [[(-1, -1, -1, 1),
#                         (-1, -1, 1, -1),
#                         (1/3, 1, 1, 1),
#                         (-1, 1, -1+2/3*(1-boundary_vf[0][0]), 1),
#                         (-1/3+2/3*(boundary_vf[0][0]), 1, 1/3, 1),
#                         (1, -1, 1, -boundary_vf[0][-1]),
#                         (1, boundary_vf[0][-1], 1, 1)]]

#     return basic_points, outer_count, curve_definitions, r, forbidden_edges


# if __name__ == "__main__":
#     start_time = time.time()
#     r_filter = 2
#     radius = 0.7
#     lower_boundary_vf = [[0.4, 0.4], [0.4, 0.4]]
#     upper_boundary_vf = [[0.6, 0.6], [0.6, 0.6]]
#     # Randomly choose boundary values between lower and upper bounds
#     boundary_vf = [[random.uniform(lower_boundary_vf[i][j], upper_boundary_vf[i][j]) for j in range(len(lower_boundary_vf[i]))] for i in range(len(lower_boundary_vf))]
#     lower_vf = [1, 0.4, 0.4, 1]
#     upper_vf = [1, 0.6, 0.6, 1]
#     # vf randomly chosen between lower and upper bounds
#     dimension = 2
#     vf = [[random.uniform(lower_vf[i], upper_vf[i]) for i in range(len(lower_vf))] for _ in range(dimension)]
#     for i in range(dimension):
#         vf[i] = [vf[i][0]] + vf[i] + [vf[i][-1]]
#     basic_points, outer_count, curve_definitions, r, forbidden_edges = cross_block(radius, vf, boundary_vf)
#     # Run the demo
#     mask = block_generation(
#         basic_points=basic_points,
#         outer_count=outer_count,
#         r=r,
#         curve_definitions=curve_definitions,
#         vf=vf,
#         forbidden_edges_set=forbidden_edges,
#         r_filter=r_filter
#     )
#     print("Time taken: {:.2f} seconds".format(time.time() - start_time))