import numpy as np
import sys
from pdb import set_trace as bp
from scipy import interpolate
from scipy.signal import tukey

## SPECFEM external mesh conversion
#folder = '../'
#sys.path.append(folder)
import LibGmsh2Specfem_convert_Gmsh_to_Specfem2D_official as conv_gmsh_to_specfem

def create_mesh_file_gmsh(file, xtopo, topo, xmin, xmax, zmin, zmax, min_size_element, max_size_element_seismic, max_size_element_atmosphere):
    
    ## Building the file line by line
    file_lines = []

    ## Corner coordinates
    xtopo = np.r_[xmin, xtopo, xmax]
    topo = np.r_[topo[0], topo, topo[-1]]

    ## Base element size
    file_lines.append( f'lc = {max(max_size_element_atmosphere, max_size_element_seismic)};\n' )

    ## Corners
    file_lines.append( f'Point(1) = {{{xmin}, {zmin}, 0, lc}};\n' )
    file_lines.append( f'Point(2) = {{{xmax}, {zmin}, 0, lc}};\n' )
    file_lines.append( f'Point(3) = {{{xmin}, {zmax}, 0, lc}};\n' )
    file_lines.append( f'Point(4) = {{{xmax}, {zmax}, 0, lc}};\n' )

    ## Topography
    file_lines.append( 'Point(5) = {{{xmin}, {z0}, 0, lc}};\n'.format(xmin=xmin, z0=topo[0]) )
    file_lines.append( 'Point(6) = {{{xmax}, {zend}, 0, lc}};\n'.format(xmax=xmax, zend=topo[-1]) )
    i_point = 7
    
    connection = ['5']
    for ii in np.arange(xtopo.size)[1:-1]:
        xx, zz = xtopo[ii], topo[ii]
        file_lines.append( f'Point({i_point}) = {{{xx}, {zz}, 0, lc}};\n' )
        connection.append(str(i_point))
        i_point += 1
    connection.append('6')
    connection = connection[::-1]
    connection = ','.join(connection)

    ## Lines between points
    file_lines.append( 'Line(1) = {1, 2};\n' )
    file_lines.append( 'Line(2) = {2, 6};\n' )
    file_lines.append( 'Line(4) = {5, 1};\n' )
    file_lines.append( 'Line(5) = {6, 4};\n' )
    file_lines.append( 'Line(6) = {4, 3};\n' )
    file_lines.append( 'Line(7) = {3, 5};\n' )
    file_lines.append( f'Line(23) = {{{connection}}};\n' )

    file_lines.append( 'Line Loop(11) = {1, 2, 23, 4};\n' )
    file_lines.append( 'Plane Surface(12) = {11};\n' ) ## Seismic

    file_lines.append( 'Line Loop(13) = {-23, 5, 6, 7};\n' )
    file_lines.append( 'Plane Surface(14) = {13};\n' ) ## Atmosphere

    file_lines.append( 'Mesh.ElementOrder = 2;\n' )

    ## Refinement strategy
    ## Seismic
    file_lines.append( 'Field[1] = MathEval;' )
    file_lines.append( f'Field[1].F = "{min_size_element} + {max_size_element_seismic-min_size_element} * abs(y)/{zmax}";' )# // Mesh size increases as -y increases (refinement decreases with depth)

    file_lines.append( 'Field[2] = Restrict;' )
    file_lines.append( 'Field[2].IField = 1;' )
    file_lines.append( 'Field[2].FacesList = {12};' )

    # Atmosphere
    file_lines.append( 'Field[3] = MathEval;' )
    file_lines.append( f'Field[3].F = "{min_size_element} + {max_size_element_atmosphere-min_size_element} * abs(y)/{zmax}";' )# // Mesh size increases as -y increases (refinement decreases with depth)

    file_lines.append( 'Field[4] = Restrict;' )
    file_lines.append( 'Field[4].IField = 3;' )
    file_lines.append( 'Field[4].FacesList = {14};' )

    # Set the background field for the whole domain
    file_lines.append( 'Field[6] = Min;' )
    file_lines.append( 'Field[6].FieldsList = {2,4};' )
    file_lines.append( 'Background Field = 6;' )

    ## Physical domain information
    file_lines.append( 'Physical Line("Top") = {6};' )
    file_lines.append( 'Physical Line("Left") = {7,4};' )
    file_lines.append( 'Physical Line("Bottom") = {1};' )
    file_lines.append( 'Physical Line("Right") = {2,5};' )

    file_lines.append( 'Physical Surface("M2") = {14};' )
    file_lines.append( 'Physical Surface("M1") = {12};' )

    file_lines.append( 'Mesh.Algorithm = 5;' )       # Use the quadrilateral mesh algorithm
    file_lines.append( 'Mesh.RecombineAll = 1;' )    # Recombine the mesh to create quadrilateral elements

    # Set the mesh format to 2.2
    file_lines.append( 'Mesh.MshFileVersion = 2.2;' )

    # Generate 2D mesh
    file_lines.append( 'Mesh 2;' )

    # Save the mesh
    file_lines.append( f'Save "{file.replace("geo", "msh")}";' )
    
    with open(file, 'w') as fp: 
        fp.write('\n'.join(file_lines))
    
    return file_lines

def fix_mesh_size(xmin, xmax, lc_g):

        #xmin, xmax = self.distance.min(), self.distance.max()
        #zmin, zmax = self.simulation_domain['z-min'], self.simulation_domain['z-max']
        lc_select = lc_g
        
        xmin_new = ((round(abs(xmin) / lc_select)+0) * lc_select)*np.sign(xmin)
        if xmin_new > xmin:
            xmin_new = ((round(abs(xmin) / lc_select)+1) * lc_select)*np.sign(xmin)
        
        xmax_new = ((round(abs(xmax) / lc_select)+0) * lc_select)*np.sign(xmax)
        if xmax_new < xmax:
            xmax_new = ((round(abs(xmax) / lc_select)+1) * lc_select)*np.sign(xmax)

        return xmin_new, xmax_new

def create_mesh_pygmsh(zmin, zmax, dists, topo, simulation_folder, lc_w, lc_g, factor_transition_zone=10., factor_pml_lc_g=10., use_cpml=True, save_mesh_file=False, alpha_taper=0.25):
    
    """
    Generate a mesh using PyGmsh and convert it to SPECFEM
    """

    import create_geom_with_pygmsh as cp
    import pygmsh
    import gmsh
    import meshio2spec2d
    gmsh.initialize()

    xmin, xmax = dists.min(), dists.max()
    xmin, xmax = fix_mesh_size(xmin, xmax, lc_g)

    L = abs(xmax-xmin)
    lc_pml = min(lc_w, lc_g)
    w_pml = lc_pml*factor_pml_lc_g
    H_t = lc_g*factor_transition_zone # width of transition layer
    nelm_h_g = int(L / lc_g) 
    nelm_h_w = int(L / lc_w) 

    lc_b = min(lc_w, lc_g) # element size at boundary
    H_w = abs(zmax)  # water depth in meter
    H_g = abs(zmin)  # subsurface depth in meter

    n_points = int(L / lc_b)
    taper = tukey(n_points, alpha=alpha_taper)
    x_arr = np.linspace(xmin, xmax, n_points)

    n_points = int(L / lc_g)
    taper_t = tukey(n_points, alpha=alpha_taper)
    x_arr_t = np.linspace(xmin, xmax, n_points)

    f = interpolate.interp1d(dists, topo, bounds_error=False, fill_value=0.,)

    #import matplotlib.pyplot as plt; plt.figure(); plt.plot(dists, topo); plt.savefig('./test_topo.png')
    #bp()

    # Initialize empty geometry using the build in kernel in GMSH
    with pygmsh.geo.Geometry() as geom:

        """ node ids without pml layer
        5             6

        4             3
        4t            3t <--- transition layer


        1             2

        """

        # points
        p1 = (xmin, -H_g, 0)
        p2 = (xmax, -H_g, 0)
        p3 = (xmax, 0, 0)
        p4 = (xmin, 0, 0)
        p5 = (xmin, H_w, 0)
        p6 = (xmax, H_w, 0)

        p3t = (xmax, 0 - H_t, 0)
        p4t = (xmin, 0 - H_t, 0)

        #
        # create subsurface geometry
        #
        topo = {"x": x_arr, "z": f(x_arr)*taper}
        topo_t = {"x": x_arr_t, "z": f(x_arr_t)*taper_t-H_t}

        # create rectangles
        whole_domain = cp.rectangles(geom)
        whole_domain.add_one_rect(geom, p1, p2, p3t, p4t, lc_g, transfinite=True, mat_tag="M1", nelm_h=nelm_h_g ,topo=topo_t)
        whole_domain.add_one_rect(geom, p4t, p3t, p3, p4, [lc_g, lc_g, lc_w, lc_w], transfinite=False, mat_tag="M1", topo=topo)
        whole_domain.add_one_rect(geom, p4, p3, p6, p5, lc_w, transfinite=True, nelm_h=nelm_h_w, mat_tag="M2")

        # create pml layer
        if use_cpml:
            whole_domain.add_pml_layers(geom, w_pml, lc_pml, top_pml=True)

        # build up edges and lines
        whole_domain.build_points_edges(geom)

        # force recombine all the surfaces
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("General.Verbosity", 5)

        # create and write mesh
        mesh = geom.generate_mesh(dim=2, order=1, verbose=False)
        if save_mesh_file:
            mesh.write(f"{simulation_folder}/mesh.msh", file_format="gmsh22")

    mio2spec = meshio2spec2d.Meshio2Specfem2D(mesh, outdir=f"{simulation_folder}/EXTMSH")
    mio2spec.write(f"extMesh")

    return xmin, xmax, nelm_h_g, nelm_h_w, zmin, zmax, topo['x'], topo['z']

##########################
if __name__ == '__main__':

    file = './test.geo'
    xtopo = np.linspace(0., 40e3, 100)
    topo = 1e3*np.sin(2*np.pi*xtopo/10e3)
    mesh = dict(
        xtopo = xtopo, 
        topo = topo, 
        xmin = xtopo.min(), 
        xmax = xtopo.max(), 
        zmin = -10.e3, 
        zmax = 10.e3
    )
    opt_gmsh = dict(
        min_size_element = 1250., 
        max_size_element_seismic = 8000., 
        max_size_element_atmosphere = 1250., 
    )
    file_lines = create_mesh_file_gmsh(file, **mesh, **opt_gmsh)

    bp()