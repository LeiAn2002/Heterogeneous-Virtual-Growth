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


def find_contours_hierarchy(bin_img):
    """
    Use OpenCV to find contours with full hierarchy info (RETR_TREE).

    Args:
        bin_img (np.ndarray): Binary image (0 or 255).

    Returns:
        contours (list): A list of contour arrays.
        hierarchy (np.ndarray): A (1, N, 4) array describing each contour's relations.
    """
    kernel = np.ones((2, 2), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
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


def write_geo_file_with_boolean_difference_and_periodic(
    contours,
    geo_filename="output.geo",
    lc=5.0,
    flip_y=True
):
    """
    Generate a Gmsh .geo file that:
      1) Creates a big bounding rectangle as outer plane surface.
      2) For each contour, define plane surface for holes.
      3) Use BooleanDifference to subtract hole surfaces from bounding surface 
         => newSurf[] = BooleanDifference{...}{...}.
      4) Coherence => edges被拆分(若孔洞贴边).
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
        f.write('Mesh.CharacteristicLengthFromPoints = 0;\n')
        f.write('Mesh.CharacteristicLengthFromCurvature = 0;\n')
        f.write('Mesh.RecombineAll = 1;\n\n')

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
        id_p1 = point_id; point_id+=1
        f.write(f'Point({point_id}) = {{{p2[0]},{fy(p2[1])},0,{lc}}};\n')
        id_p2 = point_id; point_id+=1
        f.write(f'Point({point_id}) = {{{p3[0]},{fy(p3[1])},0,{lc}}};\n')
        id_p3 = point_id; point_id+=1
        f.write(f'Point({point_id}) = {{{p4[0]},{fy(p4[1])},0,{lc}}};\n')
        id_p4 = point_id; point_id+=1

        f.write(f'Line({line_id}) = {{{id_p1},{id_p2}}};\n'); lB=line_id; line_id+=1
        f.write(f'Line({line_id}) = {{{id_p2},{id_p3}}};\n'); lR=line_id; line_id+=1
        f.write(f'Line({line_id}) = {{{id_p3},{id_p4}}};\n'); lT=line_id; line_id+=1
        f.write(f'Line({line_id}) = {{{id_p4},{id_p1}}};\n'); lL=line_id; line_id+=1

        f.write(f'Curve Loop({surf_id}) = {{{lB},{lR},{lT},{lL}}};\n')
        boundingLoopId = surf_id
        surf_id+=1
        f.write(f'Plane Surface({surf_id}) = {{{boundingLoopId}}};\n')
        boundingSurfId = surf_id
        surf_id+=1

        # (B) define hole surfaces
        holeSurfIds = []
        for i, cnt in enumerate(contours):
            npts = cnt.shape[0]
            holePointIds = []
            for j in range(npts):
                xj, yj = cnt[j,0,0], cnt[j,0,1]
                if flip_y: 
                    yj = -yj
                f.write(f'Point({point_id}) = {{{xj},{yj},0,{lc}}};\n')
                holePointIds.append(point_id)
                point_id+=1

            start_line = line_id
            for j in range(npts):
                pA = holePointIds[j]
                pB = holePointIds[(j+1) % npts]
                f.write(f'Line({line_id}) = {{{pA},{pB}}};\n')
                line_id+=1
            end_line = line_id - 1

            f.write(f'Curve Loop({surf_id}) = {{{",".join(map(str,range(start_line,end_line+1)))}}};\n')
            holeLoopId = surf_id
            surf_id+=1
            f.write(f'Plane Surface({surf_id}) = {{{holeLoopId}}};\n')
            holeSurfId = surf_id
            holeSurfIds.append(holeSurfId)
            surf_id+=1

        # (C) Boolean difference => store result in newSurf[], so we can reference it later
        f.write('\n')
        if len(holeSurfIds)>0:
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

Line finalLines[] = Boundary{{ newSurf[0] }};

leftSet[] = {{}};
rightSet[] = {{}};
bottomSet[] = {{}};
topSet[] = {{}};

tol = 1e-5;

Function xMin(l) = BoundingBox{{l}}[0];
Function yMin(l) = BoundingBox{{l}}[1];
Function xMax(l) = BoundingBox{{l}}[3];
Function yMax(l) = BoundingBox{{l}}[4];

For i In {{0 : #finalLines[]-1}}
  ll = finalLines[i];
  xm1 = xMin(ll); ym1 = yMin(ll);
  xm2 = xMax(ll); ym2 = yMax(ll);

  // left
  If( Abs(xm1 - {min_x})<tol & Abs(xm2 - {min_x})<tol )
    leftSet[#leftSet[]] = ll;
    Continue;
  EndIf
  // right
  If( Abs(xm1 - {max_x})<tol & Abs(xm2 - {max_x})<tol )
    rightSet[#rightSet[]] = ll;
    Continue;
  EndIf
  // bottom
  If( Abs(ym1 - {fy(min_y)})<tol & Abs(ym2 - {fy(min_y)})<tol )
    bottomSet[#bottomSet[]] = ll;
    Continue;
  EndIf
  // top
  If( Abs(ym1 - {fy(max_y)})<tol & Abs(ym2 - {fy(max_y)})<tol )
    topSet[#topSet[]] = ll;
    Continue;
  EndIf
EndFor

// naive pair by index
// leftSet vs rightSet
nL = #leftSet[];
nR = #rightSet[];
If(nL == nR)
  For j In {{0 : nL-1}}
    Periodic Line{{ rightSet[j] }} = {{ leftSet[j] }};
  EndFor
Else
  Printf("Warning: mismatch left(#=%g) vs right(#=%g)", nL, nR);
EndIf

// bottomSet vs topSet
nB = #bottomSet[];
nT = #topSet[];
If(nB == nT)
  For j In {{0 : nB-1}}
    Periodic Line{{ topSet[j] }} = {{ bottomSet[j] }};
  EndFor
Else
  Printf("Warning: mismatch bottom(#=%g) vs top(#=%g)", nB, nT);
EndIf

Coherence();
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
    nodes = mesh.points[:, :2].copy()
    
    cells_out = []
    for block in mesh.cells:
        ctype = block.type
        cdata = block.data
        # we only gather 2D cells: triangle or quad
        if ctype in ["triangle", "quad"]:
            cells_out.append((ctype, cdata))
    return nodes, cells_out


def generate_mesh():
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
    design_path = "./designs/2d/symbolic_graph.npy"  # your input
    geo_file = "./designs/2d/mesh.geo"
    msh_file = "./designs/2d/mesh.msh"
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
    simplified_contours = simplify_contours(contours, epsilon_ratio=0.02)

    # Step 4: Write .geo
    write_geo_file_with_boolean_difference_and_periodic(simplified_contours, geo_file, lc=50.0, flip_y=True)

    # Step 5: Run gmsh
    run_gmsh(geo_file, msh_file, dim=2)

    # Step 6: Load mesh
    # mesh = meshio.read(msh_file)

    # Here you could proceed with your homogenization or finite element routine.


generate_mesh()
