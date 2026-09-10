# convert mesh to specfem format
import numpy as np
import time
import os

try:
    import numba
except ModuleNotFoundError:
    class _NumbaFallback:
        def jit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    numba = _NumbaFallback()


def validate_quad_arrays(points, quads, max_edge_length=None):
    """
    Validate quadrilateral connectivity against node coordinates.

    Returns a report dictionary. The caller decides whether to raise, print, or
    write the report to disk.
    """

    points = np.asarray(points)[:, 0:2]
    quads = np.asarray(quads)[:, 0:4]
    quad_points = points[quads]

    shifted = np.roll(quad_points, -1, axis=1)
    signed_areas = 0.5 * np.sum(
        quad_points[:, :, 0] * shifted[:, :, 1]
        - shifted[:, :, 0] * quad_points[:, :, 1],
        axis=1,
    )
    bad_area = np.where(signed_areas <= 0.0)[0]

    edge_lengths = np.linalg.norm(quad_points - shifted, axis=2)
    max_edge_by_quad = np.max(edge_lengths, axis=1)
    if max_edge_length is None:
        bad_edges = np.array([], dtype=int)
    else:
        bad_edges = np.where(max_edge_by_quad > max_edge_length)[0]

    return {
        "points": points,
        "quads": quads,
        "quad_points": quad_points,
        "signed_areas": signed_areas,
        "edge_lengths": edge_lengths,
        "max_edge_by_quad": max_edge_by_quad,
        "bad_area": bad_area,
        "bad_edges": bad_edges,
        "max_edge_length": max_edge_length,
    }


def format_quad_validation_report(report):
    msg = [
        "Invalid quadrilateral mesh generated for SPECFEM.",
        f"  negative/zero-area quads: {report['bad_area'].size}",
        f"  overlong-edge quads: {report['bad_edges'].size}",
    ]
    if report["max_edge_length"] is not None:
        msg.append(f"  max edge length threshold: {report['max_edge_length']}")
    if report["bad_area"].size:
        i = report["bad_area"][0]
        msg.append(f"  first bad-area quad index: {i}, nodes: {report['quads'][i].tolist()}")
        msg.append(f"  first bad-area coordinates: {report['quad_points'][i].tolist()}")
    if report["bad_edges"].size:
        i = report["bad_edges"][0]
        msg.append(f"  first overlong-edge quad index: {i}, max edge: {np.max(report['edge_lengths'][i])}")
        msg.append(f"  first overlong-edge coordinates: {report['quad_points'][i].tolist()}")
    return "\n".join(msg)


def reorder_quad_nodes_counterclockwise(points, quads):
    """
    Return first-order quads ordered counter-clockwise from the lower-left node.

    Gmsh can occasionally emit valid quadrilateral vertices in a cyclic order
    that gives SPECFEM a negative Jacobian. This fixes node ordering only; it
    does not hide non-local stitching because the geometry validator still runs
    after reordering.
    """

    points = np.asarray(points)[:, 0:2]
    quads = np.asarray(quads)
    reordered = quads.copy()
    quad_points = points[quads[:, 0:4]]
    centers = np.mean(quad_points, axis=1)
    angles = np.arctan2(
        quad_points[:, :, 1] - centers[:, None, 1],
        quad_points[:, :, 0] - centers[:, None, 0],
    )
    order = np.argsort(angles, axis=1)
    sorted_quads = np.take_along_axis(quads[:, 0:4], order, axis=1)
    sorted_points = np.take_along_axis(quad_points, order[:, :, None], axis=1)

    for i in range(sorted_quads.shape[0]):
        # Use the left-most, then lowest, point as SPECFEM node 1.
        start = np.lexsort((sorted_points[i, :, 1], sorted_points[i, :, 0]))[0]
        reordered[i, 0:4] = np.roll(sorted_quads[i], -start)

    return reordered


@numba.jit(nopython=True)
def get_cpml_cells_except_damping(
                PML_X_elms_orig,
                PML_Y_elms_orig,
                PML_XY_elms_orig,
                cells_quad_total,
                cells_line_total,
                cells_line_inner_boundary,
                elm_id_offset):
    PML_X_elms = []
    PML_Y_elms = []
    PML_XY_elms = []

    # create a node list of the inner boundary
    inner_boundary_nodes = []
    for _ielm in cells_line_inner_boundary:
        inner_boundary_nodes.extend(cells_line_total[_ielm])

    for _ielm in PML_X_elms_orig:
        # offcet
        ielm = int(_ielm - elm_id_offset)
        # check if the element is not in the inner boundary
        this_nodes_elm = cells_quad_total[ielm]

        # check if the nodes are not in the inner boundary
        if_this_element_touches_inner_boundary = False
        for this_node in this_nodes_elm:
            if this_node in inner_boundary_nodes:
                if_this_element_touches_inner_boundary = True
                break

        if not if_this_element_touches_inner_boundary:
            PML_X_elms.append(_ielm)

    for _ielm in PML_Y_elms_orig:
        # offcet
        ielm = int(_ielm - elm_id_offset)
        # check if the element is not in the inner boundary
        this_nodes_elm = cells_quad_total[ielm]

        # check if the nodes are not in the inner boundary
        if_this_element_touches_inner_boundary = False
        for this_node in this_nodes_elm:
            if this_node in inner_boundary_nodes:
                if_this_element_touches_inner_boundary = True
                break

        if not if_this_element_touches_inner_boundary:
            PML_Y_elms.append(_ielm)

    for _ielm in PML_XY_elms_orig:
        # offcet
        ielm = int(_ielm - elm_id_offset)
        # check if the element is not in the inner boundary
        this_nodes_elm = cells_quad_total[ielm]

        # check if the nodes are not in the inner boundary
        if_this_element_touches_inner_boundary = False
        for this_node in this_nodes_elm:
            if this_node in inner_boundary_nodes:
                if_this_element_touches_inner_boundary = True
                break

        if not if_this_element_touches_inner_boundary:
            PML_XY_elms.append(_ielm)

    # check the changes in the number of elements
    diff_x = len(PML_X_elms_orig) - len(PML_X_elms)
    diff_y = len(PML_Y_elms_orig) - len(PML_Y_elms)
    diff_xy = len(PML_XY_elms_orig) - len(PML_XY_elms)

    print("Number of PML_X elements: ", len(PML_X_elms_orig), " -> ", len(PML_X_elms), " (", diff_x, " elements removed)")
    print("Number of PML_Y elements: ", len(PML_Y_elms_orig), " -> ", len(PML_Y_elms), " (", diff_y, " elements removed)")
    print("Number of PML_XY elements: ", len(PML_XY_elms_orig), " -> ", len(PML_XY_elms), " (", diff_xy, " elements removed)")

    return PML_X_elms, PML_Y_elms, PML_XY_elms


@numba.jit(nopython=True)
def write_str_surface(cells_line, cells_quad, bound_edges, _node_ids, flag_abs):
    """ Search element id from edge id and write to string list
        This function may be a heavy bottleneck for large meshes, as
        it is a double loop over the number of edges and elements.
        Thus it is implemented in numba to speed up the process.

        cells_line: list of all line cells
        cells_quad: list of all quad cells (elements)
        bound_edges: list of target line cells on one boundary (e.g. on top bound)
        bound_nodes: list of node ids in one single element (node ordering rule)
    """

    str_lines = []

    for line_elm in bound_edges:
        # get node ids from edge id
        nodes_ids_edge = cells_line[line_elm]
        node_0 = nodes_ids_edge[0]
        node_1 = nodes_ids_edge[1]

        # get element id and node ids from edge id
        for ielm, _node_ids_elm in enumerate(cells_quad):
            if node_0 in _node_ids_elm and node_1 in _node_ids_elm:
                break
        else:
            # If no break occurred, continue to the next iteration
            continue

        # specfem requires only 2 two edge nodes
        out_str = f"{ielm} 2 {node_0} {node_1}"

        if flag_abs is not None:
            out_str += f" {flag_abs}"

        str_lines.append(out_str)


    return str_lines



@numba.jit(nopython=True)
def get_cpml_data(mesh_sets_x, mesh_sets_y, mesh_sets_xy, cell_id_offset):

    n_elm_pml = 0
    str_lines = []

    n_elm_pml += len(mesh_sets_x)
    if (len(mesh_sets_x) > 0):
        for elm in mesh_sets_x:
            str_lines.append(str(int(elm-cell_id_offset+1)) + " " + str(1))

    n_elm_pml += len(mesh_sets_y)
    if (len(mesh_sets_y) > 0):
        for elm in mesh_sets_y:
            str_lines.append(str(int(elm-cell_id_offset+1)) + " " + str(2))

    n_elm_pml += len(mesh_sets_xy)
    if (len(mesh_sets_xy) > 0):
        for elm in mesh_sets_xy:
            str_lines.append(str(int(elm-cell_id_offset+1)) + " " + str(3))

    # add number of pml elements to the first line
    str_lines.insert(0, str(n_elm_pml))

    return str_lines


class Meshio2Specfem2D:

    # node ordering in meshio is the same as vtk
    # https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestIsoparametricCellsDemo.png

    fhead_Nodes = "Nodes"
    fhead_Mesh = "Mesh"
    fhead_Material = "Material"
    fhead_Surf_abs = "Surf_abs"
    fhead_Surf_free = "Surf_free"
    fhead_CPML = "CPML"
    fname_out = "TEST"

    fname_Nodes = ""
    fname_Mesh = ""
    fname_Material = ""
    fname_Surf_abs = ""
    fname_Surf_free = ""
    fname_CPML = ""

    n_nodes = 0
    n_cells = 0
    n_edges = 0
    if_second_order = False
    key_line = "line" # line3 for second order
    key_quad = "quad" # quad9 for second order

    # stacy boundary flags
    top_abs = False
    bot_abs = True
    left_abs = True
    right_abs = True

    use_cpml = False
    cell_id_offset = 0

    # pml transition layer
    pml_transition_layer = True


    def __init__(self, mesh, outdir = "./EXTMSH", top_abs=False, bot_abs=True, left_abs=True, right_abs=True):
        self.top_abs = top_abs
        self.bot_abs = bot_abs
        self.left_abs = left_abs
        self.right_abs = right_abs
        self.mesh = mesh
        self.outdir = outdir

        # create output directory
        import os
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # check if PML_X is included in the physical groups
        if "PML_X" in self.mesh.cell_sets_dict:
            self.use_cpml = True

        # check if second order elements are included in the mesh
        if "quad9" in self.mesh.cells_dict:
            self.if_second_order = True

        if self.if_second_order:
            self.key_line = "line3"
            self.key_quad = "quad9"
        else:
            self.key_line = "line"
            self.key_quad = "quad"


    def write_nodes(self):
        # number of nodes
        # coord x y
        nodes = self.mesh.points
        self.n_nodes = len(nodes)
        with open(self.fname_Nodes, "w") as f:
            f.write(f"{self.n_nodes}\n")
            np.savetxt(f, nodes[:,0:2], fmt="%f %f")

    def validate_quad_geometry(self, max_edge_length=None):
        """
        Fail early if the mesh contains inverted or obviously scrambled quads.

        SPECFEM will stop later on negative Jacobians, but checking here gives a
        clearer error at mesh-generation time and catches long-range node
        stitching before the external mesh files are written.
        """

        if self.key_quad not in self.mesh.cells_dict:
            raise ValueError(f"Mesh does not contain '{self.key_quad}' cells")

        points = self.mesh.points[:, 0:2]
        quads = self.mesh.cells_dict[self.key_quad][:, 0:4]
        report = validate_quad_arrays(points, quads, max_edge_length=max_edge_length)

        if report["bad_area"].size or report["bad_edges"].size:
            raise ValueError(format_quad_validation_report(report))


    def repair_quad_node_order(self):
        """
        Reorder first-order quad corner nodes into SPECFEM's expected orientation.
        """

        points = self.mesh.points[:, 0:2]
        for cell_block in self.mesh.cells:
            if cell_block.type == self.key_quad:
                cell_block.data[:, 0:4] = reorder_quad_nodes_counterclockwise(
                    points,
                    cell_block.data[:, 0:4],
                )


    def write_mesh(self):
        # number of elements
        # node id1 id2 id3 id4 (id5 id6 id7 id8 id9 for second order elements)

        with open(self.fname_Mesh, "w") as f:
            cell_data = self.mesh.cells_dict[self.key_quad] # id starts from 0
            self.n_cells = len(cell_data)
            f.write(str(self.n_cells) + "\n")
            if self.if_second_order:
                np.savetxt(f, cell_data[:,0:9], fmt="%d %d %d %d %d %d %d %d %d")
            else:
                np.savetxt(f, cell_data[:,0:4], fmt="%d %d %d %d")


    def write_material(self):

        # find all keys starting with "M" from mesh.cell_sets
        M_keys = [key for key in self.mesh.cell_sets_dict if key[0] == "M"]
        print("material keys: ", M_keys)

        # reconstruct material flag array
        arr_mflag = np.ones(self.n_cells, dtype=int) * -1

        # id offset for quad (subtract the number of lines)
        self.cell_id_offset = int(np.min(self.mesh.cell_sets_dict["M1"][self.key_quad]))

        print("cell_id_offset: ", self.cell_id_offset)

        for key in M_keys:
            for cell in self.mesh.cell_sets_dict[key][self.key_quad]:
                try:
                    icell = int(cell)-self.cell_id_offset
                    # skip the first character "M"
                    arr_mflag[icell] = int(key.lstrip("M"))
                except:
                    print(
                        "icell, cell, key, key.lstrip('M'), self.n_cells, len(arr_mflag)")
                    print(icell, cell, key, key.lstrip(
                        "M"), self.n_cells, len(arr_mflag))
                    raise Exception("Error: cell id out of range")

        # check if all cells are assigned to a material
        if -1 in arr_mflag:
            raise Exception("Error: not all cells are assigned to a material")

        # write material file
        with open(self.fname_Material, "w") as f:
            np.savetxt(f, arr_mflag, fmt="%d")


    def _write_str_surface(self, bound_edges, _node_ids, flag_abs=None):
        # pipline function to avoid passing dict to numba (unsupported)

        cells_line = self.mesh.cells_dict[self.key_line]
        cells_quad = self.mesh.cells_dict[self.key_quad]

        return write_str_surface(cells_line, cells_quad, bound_edges, _node_ids, flag_abs)


    def write_surf_free_and_abs(self):
        # free surface
        # number of free surface edges
        # elemnt id, num nodes, node id1, node id2 (, node id3 for second order elements)

        # abs boundary
        # number of abs boundary edges
        # elemnt id, num nodes, node id1, node id2 (, node id3 for second order elements), boundary flag (1=bottom, 2=right, 3=top, 4=left)

        # node id order in meshio is the same as vtk
        #
        #
        # 3      6      2
        #
        #
        # 7      8      5
        #
        #
        # 0      4      1

        bot_nodes = np.array([0, 1, 4])
        right_nodes = np.array([1, 2, 5])
        top_nodes = np.array([2, 3, 6])
        left_nodes = np.array([3, 0, 7])

        # write number of free surface edges
        edges_top = self.mesh.cell_sets_dict["Top"][self.key_line]
        edges_bot = self.mesh.cell_sets_dict["Bottom"][self.key_line]
        edges_left = self.mesh.cell_sets_dict["Left"][self.key_line]
        edges_right = self.mesh.cell_sets_dict["Right"][self.key_line]

        n_edges_free = 0
        n_edges_abs = 0

        str_lines = []

        # write number of free surface edges
        if self.bot_abs != True:
            str_lines.extend(self._write_str_surface(edges_bot, bot_nodes))
        if self.right_abs != True:
            str_lines.extend(self._write_str_surface(edges_right, right_nodes))
        if self.top_abs != True:
            str_lines.extend(self._write_str_surface(edges_top, top_nodes))
        if self.left_abs != True:
            str_lines.extend(self._write_str_surface(edges_left, left_nodes))

        n_edges_free = len(str_lines)

        # add number of free surface edges to the first line
        str_lines.insert(0, str(n_edges_free))

        np.savetxt(self.fname_Surf_free, str_lines, fmt="%s")

        # write number of abs boundary edges
        str_lines = []
        if self.bot_abs:
            str_lines.extend(self._write_str_surface(edges_bot, bot_nodes, 1))
        if self.right_abs:
            str_lines.extend(self._write_str_surface(edges_right, right_nodes, 2))
        if self.top_abs:
            str_lines.extend(self._write_str_surface(edges_top, top_nodes, 3))
        if self.left_abs:
            str_lines.extend(self._write_str_surface(edges_left, left_nodes, 4))
        n_edges_abs = len(str_lines)

        # add number of abs boundary edges to the first line
        str_lines.insert(0, str(n_edges_abs))

        np.savetxt(self.fname_Surf_abs, str_lines, fmt="%s")


    def write_cpml(self):
        """
        Write cpml data to file.
        In default, this function creates a layer of damping elements between
        the PML and the main domain.
        """

        if self.pml_transition_layer == False:
            cells_PML_X  = self.mesh.cell_sets_dict["PML_X"][self.key_quad]
            cells_PML_Y  = self.mesh.cell_sets_dict["PML_Y"][self.key_quad]
            cells_PML_XY = self.mesh.cell_sets_dict["PML_XY"][self.key_quad]
        else:
            # prepare list of cells for cpml excluding the damping layers
            _cells_inner_boundary = []

            # find keys which are _Top, _Bottom, _Left, _Right
            for key in self.mesh.cell_sets_dict:
                if key in ["_Top", "_Bottom", "_Left", "_Right"]:
                    _cells_inner_boundary.append(self.mesh.cell_sets_dict[key][self.key_line])

            # concatenate a list of numpy arrays to one numpy array
            cells_line_inner_boundary = np.concatenate(_cells_inner_boundary)

            cells_quad_total = self.mesh.cells_dict[self.key_quad]
            cells_line_total = self.mesh.cells_dict[self.key_line]

            cells_PML_X, cells_PML_Y, cells_PML_XY = get_cpml_cells_except_damping(
                self.mesh.cell_sets_dict["PML_X"][self.key_quad],
                self.mesh.cell_sets_dict["PML_Y"][self.key_quad],
                self.mesh.cell_sets_dict["PML_XY"][self.key_quad],
                cells_quad_total,
                cells_line_total,
                cells_line_inner_boundary,
                self.cell_id_offset
            )

        # n_elme pml
        # elm_id cpml_flag
        str_lines = get_cpml_data(cells_PML_X
                                , cells_PML_Y
                                , cells_PML_XY
                                , self.cell_id_offset)

        np.savetxt(self.fname_CPML, str_lines, fmt="%s")


    def write(self, filename_out="TEST", pml_transition_layer=True):

        # measure time
        start_time = time.time()

        # set output file name
        self.filename_out = filename_out

        # set pml transition layer
        self.pml_transition_layer = pml_transition_layer

        # construct file names
        self.fname_Nodes = os.path.join(self.outdir, self.fhead_Nodes + "_" + self.filename_out)
        self.fname_Mesh = os.path.join(self.outdir, self.fhead_Mesh + "_" + self.filename_out)
        self.fname_Material = os.path.join(self.outdir, self.fhead_Material + "_" + self.filename_out)
        self.fname_Surf_abs = os.path.join(self.outdir, self.fhead_Surf_abs + "_" + self.filename_out)
        self.fname_Surf_free = os.path.join(self.outdir, self.fhead_Surf_free + "_" + self.filename_out)
        self.fname_CPML = os.path.join(self.outdir, self.fhead_CPML + "_" + self.filename_out)

        # write mesh files in specfem format from meshio object
        t_start_node = time.time()
        self.write_nodes()
        t_end_node = time.time()
        print("Time elapsed for nodes: ", t_end_node - t_start_node, " seconds")

        # write mesh
        t_start_mesh = time.time()
        self.write_mesh()
        t_end_mesh = time.time()
        print("Time elapsed for mesh: ", t_end_mesh - t_start_mesh, " seconds")

        # write material
        t_start_material = time.time()
        self.write_material()
        t_end_material = time.time()
        print("Time elapsed for material: ", t_end_material - t_start_material, " seconds")

        # write free surface
        t_start_surf = time.time()
        self.write_surf_free_and_abs()
        t_end_surf = time.time()
        print("Time elapsed for surf: ", t_end_surf - t_start_surf, " seconds")

        # write cpml
        if self.use_cpml:
            t_start_cpml = time.time()
            self.write_cpml()
            t_end_cpml = time.time()
            print("Time elapsed for cpml: ", t_end_cpml - t_start_cpml, " seconds")

        print("Time elapsed: ", time.time() - start_time, " seconds")

