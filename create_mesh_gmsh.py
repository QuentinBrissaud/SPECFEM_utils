import numpy as np
import sys
from pdb import set_trace as bp

## SPECFEM external mesh conversion
folder = '/staff/quentin/Documents/Projects/2023_Celine_internship/msc_celine_specfem/utils/utils_specfem2d/Gmsh/'
sys.path.append(folder)
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