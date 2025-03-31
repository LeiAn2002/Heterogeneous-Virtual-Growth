import os
import subprocess
import cv2
import meshio
import numpy as np


def bin_array_to_cv2(bin_array, invert=False):
    """
    Convert a 2D numpy array of 0/1 to a 0/255 uint8 image for OpenCV.
    If invert=True, then 1->0, 0->255 (flip black/white).
    Otherwise 1->255, 0->0.
    
    Args:
        bin_array (np.ndarray): shape (H, W) with values in {0,1}.
        invert (bool): whether to invert black/white.
    
    Returns:
        bin_img (np.ndarray): shape (H, W), uint8 with 0/255
    """
    if invert:
        # 1 => 0,  0 => 255
        bin_img = np.where(bin_array>0, 0, 255).astype(np.uint8)
    else:
        # 1 => 255,  0 => 0
        bin_img = np.where(bin_array>0, 255, 0).astype(np.uint8)
    return bin_img


design_path = "designs/2d/symbolic_graph.npy"
debug_prefix = "designs/2d/mesh_debug/"
design_matrix = np.load(design_path)
bin_img = bin_array_to_cv2(design_matrix, invert=True)


def find_contours_hierarchy(bin_img):
    """
    Use OpenCV to find contours with full hierarchy info (RETR_TREE).

    Args:
        bin_img (np.ndarray): Binary image (0 or 255).

    Returns:
        contours (list): A list of contour arrays.
        hierarchy (np.ndarray): A (1, N, 4) array describing each contour's relations.
    """
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # debug_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(debug_img, contours, -1, (0,0,255), 2)
    # cv2.imwrite(debug_prefix + "contours.png", debug_img)
    return contours, hierarchy


def simplify_contours(contours, epsilon_ratio=0.01):
    """
    Simplify each contour using cv2.approxPolyDP to reduce the number of points.

    Args:
        contours (list): List of contour arrays from OpenCV.
        epsilon_ratio (float): Ratio of arcLength used for simplification.

    Returns:
        new_contours (list): List of simplified contour arrays.
    """
    new_contours = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        new_contours.append(approx)

    # debug_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(debug_img, new_contours, -1, (0,0,255), 2)
    # cv2.imwrite(debug_prefix + "new_contours.png", debug_img)

    return new_contours


def get_nesting_level(idx, hierarchy):
    """
    Compute nesting level of a contour by ascending parent links.
    Even level => outer boundary, odd => hole, etc.
    """
    level = 0
    parent = hierarchy[0, idx, 3]
    while parent != -1:
        level += 1
        parent = hierarchy[0, parent, 3]
    return level


def write_geo_file_with_bounding_box(
    contours,
    geo_filename="output.geo",
    lc=5.0,
    flip_y=True,
    margin=1e-5
):
    """
    Write a Gmsh .geo file (Plan #1: one bounding rectangle as outer boundary, 
    plus holes). Then add Periodic lines for left-right and bottom-top edges.
    
    We'll force the rectangle edges to be lines:
       l1 = bottom, l2 = right, l3 = top, l4 = left
    so we can do:
       Periodic Line{l2} = {l4};  // right = left
       Periodic Line{l3} = {l1};  // top   = bottom
       Coherence;
    
    Args:
        contours (list): List of hole contours from cv2.findContours ( Nx1x2 points each ).
        geo_filename (str): Output .geo file path.
        lc (float): Characteristic length for Gmsh.
        flip_y (bool): Whether to invert y coords (OpenCV to typical CAD).
        margin (float): Small margin to avoid holes exactly on boundary 
                        which can cause gmsh geometry overlap.
    """
    # 1) Determine bounding box from all contours
    all_x, all_y = [], []
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            all_x.append(x)
            all_y.append(y)
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Expand by margin
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    
    # 2) Open .geo file
    with open(geo_filename, 'w') as f:
        f.write('// Gmsh geometry with bounding-box outer boundary + holes + PERIODIC\n')
        f.write('SetFactory("OpenCASCADE");\n\n')
        
        point_id = 1
        line_id = 1
        loop_id = 1
        
        # We'll store hole loops in hole_loop_ids
        hole_loop_ids = []
        
        # Helper to flip y
        def flipy(val):
            return -val if flip_y else val
        
        # ---------------------------------------------------------
        # (A) Define bounding rectangle:
        #     p1->p2 (Line l1) = bottom
        #     p2->p3 (Line l2) = right
        #     p3->p4 (Line l3) = top
        #     p4->p1 (Line l4) = left
        # ---------------------------------------------------------
        p1 = (min_x, min_y)
        p2 = (max_x, min_y)
        p3 = (max_x, max_y)
        p4 = (min_x, max_y)
        
        # Write rectangle points
        f.write(f'Point({point_id}) = {{{p1[0]}, {flipy(p1[1])}, 0, {lc}}};\n')
        id_p1 = point_id; point_id += 1
        f.write(f'Point({point_id}) = {{{p2[0]}, {flipy(p2[1])}, 0, {lc}}};\n')
        id_p2 = point_id; point_id += 1
        f.write(f'Point({point_id}) = {{{p3[0]}, {flipy(p3[1])}, 0, {lc}}};\n')
        id_p3 = point_id; point_id += 1
        f.write(f'Point({point_id}) = {{{p4[0]}, {flipy(p4[1])}, 0, {lc}}};\n')
        id_p4 = point_id; point_id += 1
        
        # Lines
        # l1: bottom, l2: right, l3: top, l4: left
        f.write(f'Line({line_id}) = {{{id_p1}, {id_p2}}};\n')  # bottom
        l1 = line_id
        line_id += 1
        
        f.write(f'Line({line_id}) = {{{id_p2}, {id_p3}}};\n')  # right
        l2 = line_id
        line_id += 1
        
        f.write(f'Line({line_id}) = {{{id_p3}, {id_p4}}};\n')  # top
        l3 = line_id
        line_id += 1
        
        f.write(f'Line({line_id}) = {{{id_p4}, {id_p1}}};\n')  # left
        l4 = line_id
        line_id += 1
        
        # Outer loop
        f.write(f'Curve Loop({loop_id}) = {{{l1},{l2},{l3},{l4}}};\n')
        outer_loop_id = loop_id
        loop_id += 1
        
        # ---------------------------------------------------------
        # (B) Holes: each contour => lines => hole loop
        # ---------------------------------------------------------
        for i, cnt in enumerate(contours):
            n_pts = cnt.shape[0]
            contour_pts = []
            for j in range(n_pts):
                xj, yj = cnt[j, 0, 0], cnt[j, 0, 1]
                if flip_y:
                    yj = -yj
                f.write(f'Point({point_id}) = {{{xj},{yj},0,{lc}}};\n')
                contour_pts.append(point_id)
                point_id += 1
            
            start_line = line_id
            for j in range(n_pts):
                pA = contour_pts[j]
                pB = contour_pts[(j+1)%n_pts]
                f.write(f'Line({line_id}) = {{{pA},{pB}}};\n')
                line_id += 1
            end_line = line_id - 1
            
            current_loop_id = loop_id
            lines_str = ",".join(str(lid) for lid in range(start_line, end_line+1))
            f.write(f'Curve Loop({current_loop_id}) = {{{lines_str}}};\n')
            hole_loop_ids.append(current_loop_id)
            loop_id += 1
        
        # ---------------------------------------------------------
        # (C) Single plane surface => outer_loop + holes
        # ---------------------------------------------------------
        plane_surface_id = 1
        if len(hole_loop_ids) > 0:
            hole_str = ",".join(str(hid) for hid in hole_loop_ids)
            f.write(f'Plane Surface({plane_surface_id}) = {{{outer_loop_id},{hole_str}}};\n')
        else:
            f.write(f'Plane Surface({plane_surface_id}) = {{{outer_loop_id}}};\n')
        
        # ---------------------------------------------------------
        # (D) Add PERIODIC for lines: l2=right, l4=left, l3=top, l1=bottom
        #     => we want left-right periodic, top-bottom periodic
        #     => in Gmsh syntax: 
        #         Periodic Line{ RightLineID } = { LeftLineID };
        #         Periodic Line{ TopLineID   } = { BottomLineID };
        #         Coherence;
        #    Make sure your geometry REALLY matches l2-l4 horizontally, l1-l3 vertically
        #    i.e. the bounding box is perfect rectangle. 
        # ---------------------------------------------------------
        
        f.write(f"Periodic Line{{{l2}}} = {{{l4}}};\n")  # right = left
        f.write(f"Periodic Line{{{l3}}} = {{{l1}}};\n")  # top   = bottom
        f.write("Coherence;\n")
        
        f.write('// end .geo\n')
    
    print(f"[+] Wrote periodic .geo => {geo_filename}")


def run_gmsh(geo_file, msh_file="/design/2d/fem_mesh.msh", dim=2):
    """
    Simple Gmsh invocation: .geo => .msh
    Because the .geo file now includes all bounding and Periodic commands.
    """
    cmd = ["gmsh", geo_file, f"-{dim}", "-o", msh_file]
    print("[Cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[+] Generated mesh file with periodic edges => {msh_file}")


contours, hierarchy = find_contours_hierarchy(bin_img)
simplified_contours = simplify_contours(contours, epsilon_ratio=0.01)
geo_file = debug_prefix + "output.geo"
write_geo_file_with_bounding_box(
    simplified_contours,
    geo_filename=geo_file,
    lc=5.0,
    flip_y=True,
    margin=1e-5
)
run_gmsh(geo_file, msh_file=debug_prefix + "output.msh", dim=2)


def load_msh_with_meshio(msh_file):
    """
    Load the .msh file with meshio, return (nodes, cells).
    
    Args:
        msh_file (str): Path to .msh file
    
    Returns:
        nodes (np.ndarray): shape (N, 2), node coords
        cells (list): list of (cell_type, indices_array)
    """
    mesh = meshio.read(msh_file)
    # mesh.points -> (N, 3)
    nodes = mesh.points[:, :2].copy()
    
    cells_out = []
    for block in mesh.cells:
        ctype = block.type
        cdata = block.data
        # we only gather 2D cells: triangle or quad
        if ctype in ["triangle", "quad"]:
            cells_out.append((ctype, cdata))
    return nodes, cells_out


# def main():
#     """
#     Full pipeline demonstration.
#     1. Preprocess image -> binary
#     2. Find contours+hierarchy
#     3. Simplify polygons
#     4. Build surfaces with holes
#     5. Write .geo
#     6. Run gmsh
#     7. Load .msh
#     """
#     image_path = "your_periodic_tile.png"  # your input
#     geo_file   = "output.geo"
#     msh_file   = "output.msh"
    
#     # Step 1: Preprocess
#     bin_img = preprocess_image(
#         image_path, 
#         threshold_val=127, 
#         morph_open_ksize=3, 
#         morph_close_ksize=3, 
#         blur_ksize=3, 
#         invert=False
#     )
    
#     # Step 2: Find contours
#     contours, hierarchy = find_contours_hierarchy(bin_img)
#     if hierarchy is None:
#         # means no contour found
#         print("No contours found. Exiting.")
#         return
    
#     # Step 3: Simplify polygons
#     simplified_contours = simplify_contours(contours, epsilon_ratio=0.01)
    
#     # Step 4: Build surfaces with holes
#     surfaces = build_surfaces_with_holes(simplified_contours, hierarchy)
    
#     # Step 5: Write .geo
#     write_geo_file(simplified_contours, surfaces, geo_filename=geo_file, lc=10.0, flip_y=True)
    
#     # Step 6: Run gmsh
#     run_gmsh(geo_file, msh_file, dim=2)
    
#     # Step 7: Load mesh
#     nodes, cells = load_msh_with_meshio(msh_file)
    
#     print(f"Mesh loaded: #nodes={nodes.shape[0]}, #cell_blocks={len(cells)}")
#     total_elems = sum(bl[1].shape[0] for bl in cells)
#     print(f"Total elements = {total_elems}")
    
#     # Here you could proceed with your homogenization or finite element routine.

# if __name__ == "__main__":
#     main()