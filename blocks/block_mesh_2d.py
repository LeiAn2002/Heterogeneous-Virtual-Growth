import os
import subprocess
import cv2
import meshio
import numpy as np
import math


def is_on_boundary(px, py, min_x, max_x, min_y, max_y, tol=1e-12):
    """
    Check if (px,py) is on the rectangular boundary [min_x, max_x] x [min_y, max_y],
    within a small numerical tolerance 'tol'.
    """
    on_left   = abs(px - min_x) < tol
    on_right  = abs(px - max_x) < tol
    on_bottom = abs(py - min_y) < tol
    on_top    = abs(py - max_y) < tol
    return (on_left or on_right or on_bottom or on_top)


def distance_to_rect(px, py, min_x, max_x, min_y, max_y):
    """
    Compute minimum distance from point(px, py) to the rectangle edges
    defined by [min_x, max_x] x [min_y, max_y].
    """
    d_left   = abs(px - min_x)
    d_right  = abs(px - max_x)
    d_bottom = abs(py - min_y)
    d_top    = abs(py - max_y)
    return min(d_left, d_right, d_bottom, d_top)


def filter_close_points(cnt, min_x, max_x, min_y, max_y, min_dist=2.0, boundary_tol=1e-5):
    """
    Given a contour (list/array of points), merge any two consecutive points
    whose distance < min_dist.
    
    Implementation detail:
    - We'll do a single pass from i=0 to i=len(cnt)-1.
    - If dist(cnt[i], cnt[i+1]) < min_dist, 
      we create newPt = midpoint(cnt[i], cnt[i+1]),
      add newPt to filtered, then skip the next index.
    - Otherwise we keep cnt[i].
    - At the end, we also handle the last point if it wasn't merged.
    
    If the contour is closed (the last point connects back to the first),
    you can do more advanced wrap-around logic. Here we'll keep it simple:
    no wrap-around merging.
    """
    filtered = []
    i = 0
    n = len(cnt)
    while i < n:
        if i == n - 1:
            # last point => just keep it
            filtered.append(cnt[i])
            i += 1
        else:
            # check dist of current point vs next
            pA = cnt[i][0]  # shape (2,)
            pB = cnt[i+1][0]
            dx = pA[0] - pB[0]
            dy = pA[1] - pB[1]
            distAB = math.hypot(dx, dy)

            onA = is_on_boundary(pA[0], pA[1], min_x, max_x, min_y, max_y, boundary_tol)
            onB = is_on_boundary(pB[0], pB[1], min_x, max_x, min_y, max_y, boundary_tol)

            if distAB < min_dist and not (onA and onB):
                # measure distance to bounding rect
                dA = distance_to_rect(pA[0], pA[1], min_x, max_x, min_y, max_y)
                dB = distance_to_rect(pB[0], pB[1], min_x, max_x, min_y, max_y)
                if dA <= dB:
                    # A is closer => keep A
                    filtered.append(cnt[i])
                else:
                    # B is closer => keep B
                    filtered.append(cnt[i+1])
                i += 2  # skip next
            else:
                # keep the current point
                filtered.append(cnt[i])
                i += 1

    return filtered


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


def find_contours_hierarchy(bin_img):
    """
    Use OpenCV to find contours with full hierarchy info (RETR_TREE).

    Args:
        bin_img (np.ndarray): Binary image (0 or 255).

    Returns:
        contours (list): A list of contour arrays.
        hierarchy (np.ndarray): A (1, N, 4) array describing each contour's relations.
    """
    # kernel_open = np.ones((3, 3), np.uint8)
    # kernel_close = np.ones((3, 3), np.uint8)
    # bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_open, iterations=1)
    # bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    debug_prefix = "./designs/2d/"
    debug_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(debug_prefix + "contours_for_debug.png", debug_img)
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


def write_geo_file_with_boolean_difference_and_periodic(
    contours,
    geo_filename="output.geo",
    lc=5.0,
    flip_y=True,
    min_dist_for_merge=2.0
):
    """
    Generate a Gmsh .geo file that:
      1) Creates a big bounding rectangle as outer plane surface.
      2) For each contour, define plane surface for holes.
      3) Use BooleanDifference to subtract hole surfaces from bounding surface 
         => newSurf[] = BooleanDifference{...}{...}.
      4) Coherence => edges
      5) Then we gather final boundary lines with "Line finalLines[] = Boundary{newSurf[0]};"
         and script a loop to classify each line => left/right/bottom/top, 
         then do line-by-line Periodic pair.

    This avoids "Surface In BoundingBox(...)" syntax, so older Gmsh won't error out.

    Note: If holes heavily subdivide edges, you might want to 
          sort lines by mid coordinate for more accurate pairing.

    Args:
        contours (list): 2D hole contours (e.g. from cv2.findContours).
        geo_filename (str):  path to .geo file to be written.
        lc (float): characteristic length used in Gmsh for meshing.
        flip_y (bool): if True, invert y-coordinates from (OpenCV to typical CAD).
    """

    # 1) find bounding box from all contour points
    all_x, all_y = [], []
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            all_x.append(x)
            all_y.append(y)
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    with open(geo_filename, 'w') as f:
        f.write('// Gmsh geometry with boolean difference + line-by-line periodic\n')
        f.write('SetFactory("OpenCASCADE");\n\n')

        # optional mesh constraints
        f.write('Mesh.CharacteristicLengthMax = 10;\n')
        f.write('Mesh.CharacteristicLengthMin = 1;\n')
        # f.write('Mesh.CharacteristicLengthExtendFromBoundary = 1;\n')
        f.write('Mesh.CharacteristicLengthFromPoints = 0;\n')
        f.write('Mesh.CharacteristicLengthFromCurvature = 0;\n')
        f.write('Mesh.RecombineAll = 1;\n\n')
        f.write('Mesh.RecombinationAlgorithm = 5;\n\n')
        f.write('Mesh.ElementOrder = 0;\n\n')

        def fy(v):
            return -v if flip_y else v

        point_id = 1
        line_id = 1
        surf_id = 1

        # (A) bounding rectangle => plane surface
        p1 = (min_x, min_y)
        p2 = (max_x, min_y)
        p3 = (max_x, max_y)
        p4 = (min_x, max_y)

        f.write(f'Point({point_id}) = {{{p1[0]},{fy(p1[1])},0,{lc}}};\n')
        id_p1 = point_id
        point_id += 1
        f.write(f'Point({point_id}) = {{{p2[0]},{fy(p2[1])},0,{lc}}};\n')
        id_p2 = point_id
        point_id += 1
        f.write(f'Point({point_id}) = {{{p3[0]},{fy(p3[1])},0,{lc}}};\n')
        id_p3 = point_id
        point_id += 1
        f.write(f'Point({point_id}) = {{{p4[0]},{fy(p4[1])},0,{lc}}};\n')
        id_p4 = point_id
        point_id += 1

        f.write(f'Line({line_id}) = {{{id_p1},{id_p2}}};\n')
        lB = line_id
        line_id += 1
        f.write(f'Line({line_id}) = {{{id_p2},{id_p3}}};\n')
        lR = line_id
        line_id += 1
        f.write(f'Line({line_id}) = {{{id_p3},{id_p4}}};\n')
        lT = line_id
        line_id += 1
        f.write(f'Line({line_id}) = {{{id_p4},{id_p1}}};\n')
        lL = line_id
        line_id += 1

        f.write(f'Curve Loop({surf_id}) = {{{lB},{lR},{lT},{lL}}};\n')
        boundingLoopId = surf_id
        surf_id += 1
        f.write(f'Plane Surface({surf_id}) = {{{boundingLoopId}}};\n')
        boundingSurfId = surf_id
        surf_id += 1

        # (B) define hole surfaces
        # min_dist_for_merge = max(abs(min_x), abs(max_x)) * 0.01
        holeSurfIds = []
        for i, cnt in enumerate(contours):
            for i in range(4):  # range 4 because we have quadrilateral elements
                cnt = filter_close_points(cnt, min_x, max_x, min_y, max_y, min_dist=min_dist_for_merge)

            npts = len(cnt)
            if npts < 2:
                # If it becomes too short, skip
                continue

            holePointIds = []
            for j in range(npts):
                xj, yj = cnt[j][0]
                if flip_y:
                    yj = -yj
                f.write(f'Point({point_id}) = {{{xj},{yj},0,{lc}}};\n')
                holePointIds.append(point_id)
                point_id += 1

            start_line = line_id
            for j in range(npts):
                pA = holePointIds[j]
                pB = holePointIds[(j+1) % npts]  # wrap around if closed
                f.write(f'Line({line_id}) = {{{pA},{pB}}};\n')
                line_id += 1
            end_line = line_id-1

            f.write(f'Curve Loop({surf_id}) = {{{",".join(map(str,range(start_line,end_line+1)))}}};\n')
            holeLoopId = surf_id
            surf_id += 1
            f.write(f'Plane Surface({surf_id}) = {{{holeLoopId}}};\n')
            holeSurfId = surf_id
            holeSurfIds.append(holeSurfId)
            surf_id += 1

        # (C) Boolean difference => store result in newSurf[], so we can reference it later
        f.write('\n')
        if len(holeSurfIds) > 0:
            holes_str = ",".join(str(h) for h in holeSurfIds)
            f.write('newSurf[] = BooleanDifference{ Surface{')
            f.write(str(boundingSurfId))
            f.write('}; Delete; }{ Surface{')
            f.write(holes_str)
            f.write('}; Delete; };\n')
        else:
            # no holes => just define newSurf[0] as boundingSurfId
            f.write(f'newSurf[] = {{{boundingSurfId}}};\n')

        f.write('Coherence;\n\n')

        # (D) Insert script to gather boundary lines of newSurf[0], then do classification & Periodic
        # We'll define Xmin=..., Xmax=..., Ymin=..., Ymax=...
        # Then use bounding box logic. 
        # Must carefully handle sign if flip_y. Let's keep direct bounding coords:

        bounding_script = f"""
//////////////////////////////////////////////////////
// (D) Classify final boundary lines of newSurf[0], 
// then do line-by-line periodic
//////////////////////////////////////////////////////

finalLines[] = Boundary{{ Surface{{newSurf[0]}}; }};

leftSet[] = {{}};
rightSet[] = {{}};
bottomSet[] = {{}};
topSet[] = {{}};

tol = 1e-5;

For i In {{0 : #finalLines[]-1}}
  ll = finalLines[i];
  bb[] = BoundingBox Line {{ll}};
  xm1 = bb[0];
  ym1 = bb[1];
  xm2 = bb[3];
  ym2 = bb[4];

  // left
  If( Abs(xm1 - {min_x})<tol && Abs(xm2 - {min_x})<tol )
    leftSet[#leftSet[]] = ll;
  EndIf
  // right
  If( Abs(xm1 - {max_x})<tol && Abs(xm2 - {max_x})<tol )
    rightSet[#rightSet[]] = ll;
  EndIf
  // bottom
  If( Abs(ym1 - {fy(min_y)})<tol && Abs(ym2 - {fy(min_y)})<tol )
    bottomSet[#bottomSet[]] = ll;
  EndIf
  // top
  If( Abs(ym1 - {fy(max_y)})<tol && Abs(ym2 - {fy(max_y)})<tol )
    topSet[#topSet[]] = ll;
  EndIf
EndFor

// naive pair by index
// leftSet vs rightSet
nL = #leftSet[];
nR = #rightSet[];
If(nL == nR)
  For j In {{0 : nL-1}}
    reverseIndex = nL - 1 - j;
    Periodic Line{{ rightSet[j] }} = {{ leftSet[reverseIndex] }};
  EndFor
Else
  Printf("Warning: mismatch left(#=%g) vs right(#=%g)", nL, nR);
EndIf

// bottomSet vs topSet
nB = #bottomSet[];
nT = #topSet[];
If(nB == nT)
  For j In {{0 : nB-1}}
    reverseIndex = nB - 1 - j;
    Periodic Line{{ topSet[j] }} = {{ bottomSet[reverseIndex] }};
  EndFor
Else
  Printf("Warning: mismatch bottom(#=%g) vs top(#=%g)", nB, nT);
EndIf

Coherence;
//////////////////////////////////////////////////////
"""
        f.write(bounding_script)
        f.write("// end final script\n")

    print(f"[+] Wrote .geo with advanced boolean diff + line-by-line periodic =>", geo_filename)


def run_gmsh(geo_file, msh_file="/design/2d/fem_mesh.msh", dim=2):
    """
    Simple Gmsh invocation: .geo => .msh
    Because the .geo file now includes all bounding and Periodic commands.
    """
    cmd = ["gmsh", geo_file, f"-{dim}", "-o", msh_file]
    print("[Cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[+] Generated mesh file with periodic edges => {msh_file}")


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
    nodes = mesh.points.copy()

    tri_elems = []
    quad_elems = []
    for block in mesh.cells:
        if block.type == "triangle":
            tri_elems.append(block.data)
        if block.type == "quad":
            quad_elems.append(block.data)
    if len(tri_elems) > 0:
        tri_elems = np.vstack(tri_elems)
        tri_elems = tri_elems[:, [0, 2, 1]]
    else:
        tri_elems = np.zeros((0, 3), dtype=int)
    if len(quad_elems) > 0:
        quad_elems = np.vstack(quad_elems)
        quad_elems = quad_elems[:, [0, 3, 2, 1]]
    else:
        quad_elems = np.zeros((0, 4), dtype=int)
    return nodes, tri_elems, quad_elems


def downscale_binary_image(bin_img, 
                           target_size=(None, None), 
                           scale=0.25, 
                           interpolation=cv2.INTER_AREA):
    """
    Downscale a large binary image (0 or 255) to a smaller size 
    while trying to preserve shape outlines.

    Args:
        bin_img (np.ndarray): Original large binary image, shape (H,W), dtype=uint8 in {0,255}
        target_size (tuple): (newWidth, newHeight); if both not None, we use it directly
        scale (float): if target_size is None, we compute new size = scale * original
        interpolation: e.g. cv2.INTER_AREA for downscaling

    Returns:
        small_bin (np.ndarray): The smaller binary image (0/255)
    """
    H, W = bin_img.shape[:2]

    # 1) Decide new size
    if target_size[0] is not None and target_size[1] is not None:
        newW, newH = target_size
    else:
        newW = int(W * scale)
        newH = int(H * scale)

    # 2) Resize
    # Convert to float or keep as is? We can keep as uint8 because we're simply downscaling
    # bin_img is already 0/255, so direct usage is okay
    resized = cv2.resize(bin_img, (newW, newH), interpolation=interpolation)

    # 3) Optional threshold => ensure it's strictly 0 or 255
    # Because resize might produce values in [0..255] but not strictly 0 or 255
    # if using INTER_AREA or something else
    # If we want a strict bin:
    _, small_bin = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    return small_bin


def generate_mesh(
        design_path="./designs/2d/symbolic_graph.npy",
        geo_file="./designs/2d/mesh.geo",
        msh_file="./designs/2d/mesh.msh"
        ):
    """
    Full pipeline demonstration.
    1. Preprocess image -> binary
    2. Find contours+hierarchy
    3. Simplify polygons
    4. Build surfaces with holes
    5. Write .geo
    6. Run gmsh
    7. Load .msh
    """
    bin_array = np.load(design_path)

    # Step 1: Preprocess
    bin_img = bin_array_to_cv2(bin_array, invert=True)

    # Step 2: Find contours
    contours, hierarchy = find_contours_hierarchy(bin_img)
    if hierarchy is None:
        # means no contour found
        print("No contours found. Exiting.")
        return

    # Step 3: Simplify polygons
    # simplified_contours = simplify_contours(contours, epsilon_ratio=0.00)

    # Step 4: Write .geo
    write_geo_file_with_boolean_difference_and_periodic(contours, geo_file, lc=20.0, flip_y=True, min_dist_for_merge=8)

    # Step 5: Run gmsh
    run_gmsh(geo_file, msh_file, dim=2)

    # downscaled_bin_img = downscale_binary_image(bin_img, scale=0.5, interpolation=cv2.INTER_AREA)
    # cv2.imwrite("./designs/2d/downscaled.png", downscaled_bin_img)
    # print("Downscaled shape:", downscaled_bin_img.shape[0] * downscaled_bin_img.shape[1])

    # Step 6: Load mesh
    # mesh = meshio.read(msh_file)

    # Here you could proceed with your homogenization or finite element routine.


if __name__ == "__main__":
    # Run the full pipeline
    generate_mesh()
