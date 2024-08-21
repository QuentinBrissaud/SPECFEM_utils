#!/usr/bin/env python3
from pdb import set_trace as bp
import os 
import pandas as pd
import numpy as np
import sys
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
import fluids
from pyrocko import moment_tensor as mtm
import pickle

## Custom libraries
import construct_atmospheric_model, collect_topo, create_mesh_gmsh

try:
    import collect_ECMWF
    imported_ECMWF = True
except:
    print('Can not find collect_ECMWF')
    imported_ECMWF = True
    
def write_params(file, params):

    """
    Write SPECFEM input parameter file
    """
    
    ## Open file and temporary file
    file_r = open(file, 'r')
    lines  = file_r.readlines()
    
    temp = file +'_temp'
    file_w = open(temp, 'w')

    ## Loop over each file line
    list_idx = params.index.tolist()
    cpt_writing = 0
    for line in lines:
    
        list_bool = [True if idx.strip() == line.split('=')[0].strip() else False for idx in list_idx]
        
        if cpt_writing > 0:
            cpt_writing -= 1
            file_w.write(lines_to_write[len(lines_to_write) - cpt_writing - 1] + '\n')
            continue
        
        if(True in list_bool and not line[0] == '#'):
        
            try:
                line_w = line.split('=')[1].split('#')[0]
                break_line = '\n' if not '#' in line else ''
                file_w.write(line.replace(line_w, ' ' + str(params.iloc[list_bool.index(True)].values[0]) + break_line))
            except:
                file_w.write(line) ## Lines of numbers are not changed
        else:
        
            ## Write physical domain in parfile
            if 'Set the different regions and model number for each region' in line:
                lines_to_write = params.loc['domain'].values[0].split('\n')
                cpt_writing = len(lines_to_write)
                continue
            file_w.write(line)
            
    ## Close files
    file_r.close()
    file_w.close()
    
    os.system('mv ' + temp + ' ' + file)
    
def load_params(file):

    """
    Load SPECFEM input parameter file
    """

    params = pd.read_csv(file, engine='python', delimiter=' *= *', comment='#', skiprows=[0,1,2,3,8,12])
    params.columns = ['param', 'value']
    params.set_index('param', inplace=True)
    return params
 
def load_stations(file):

    """
    Load SPECFEM station file
    """
    
    stations = pd.read_csv(file, engine='python', delim_whitespace=True, header=None)
    stations.columns = ['name', 'array', 'x', 'z', 'd0', 'd1']
    return stations
 
def compute_distance(x, source_latlon):

    """
    Compute cartesian distance between a source and a station
    """

    if x['coordinates'] == 'latlon':
        distance = gps2dist_azimuth(source_latlon[0], source_latlon[1], x['lat'], x['lon'])
        x['x']   = distance[0]
        x['baz'] = distance[2]
    else: # default case
        x['lat'] = source_latlon[0]
        x['lon'] = source_latlon[1] + kilometer2degrees(x['x'])
        x['baz'] = 0.
        
    return x
 
def correct_coordinates_stations(input_station, source_latlon):

    """
    Convert latitude/longitude coordinates into SPECFEM-readable cartesian coordinates
    """
    
    filter = input_station.loc[(input_station['coordinates'] == 'latlon'), :]
    if source_latlon:
        if filter.size > 0:
            if filter['lat'].isnull().values.any():
                sys.exit("latitude and longitude needs to be provided if 'coordinates' == 'latlon'.")
        
    else:
        if filter.size > 0:
            sys.exit('Please provide source latitude and longitude to use lat/lon station coordinates.')
    
    input_station = input_station.apply(compute_distance, args=[source_latlon], axis=1)
        
    return input_station
 
def build_stations(stations, input_station):

    """
    Convert user-provided station dictionnary into DataFrame
    """
    
    default_entry = {}
    for key in stations.columns:
        default_entry[key] = '0'
    
    new_stations = pd.DataFrame(columns=stations.columns)
    for _, station in input_station.iterrows():
        new_entry = default_entry.copy()
        for key in station.keys():
            new_entry[key] = station[key]
        new_stations = new_stations.append( [new_entry] )
    
    new_stations.reset_index(drop=True, inplace=True)
    
    return new_stations
 
def create_station_file(simulation):

    """
    Create SPECFEM station file from a station DataFrame
    """

    stations = simulation.stations
    file     = simulation.station_file
    format_station = ['name', 'array', 'x', 'z', 'd0', 'd1']
    stations[format_station].to_csv(file, sep='\t', header=False, index=False)
 
def get_name_new_folder(dir_new_folder, template):

    """
    Get name of new simulation folder based on template and existing simulation folders
    """

    max_no = -1
    for  subdir, _, _ in os.walk(dir_new_folder):
        try:
            no = int(subdir.split('_')[-1])
        except:
            continue
        
        if not subdir.split('/')[-1] == template.format(no=str(no)):
            continue
        
        max_no = max(no, max_no)
        
    return dir_new_folder + template.format(no=str(max_no+1))
        
def copy_sample(dir_new_folder, dir_sample, template):

    """
    Copy sample simulation folder to get new simulation folder
    """
        
    new_folder = get_name_new_folder(dir_new_folder, template)
    loc_dict = {
        'sample': dir_sample,
        'new_folder': new_folder,
    }
    cmd = 'cp -R {sample} {new_folder}'
    os.system(cmd.format(**loc_dict))
    
    return new_folder
 
def update_params(params, input_dict):

    """
    Update SPECFEM parameter file based on user-provided input parameters
    """

    dict_conversion = {
        True: '.true.',
        False: '.false.',
    }
    for key in input_dict:
        if key == 'date':
            continue
        
        if isinstance(input_dict[key], pd.DataFrame):
            continue
        elif input_dict[key] in dict_conversion and not isinstance(input_dict[key], float):
            params.loc[key] = dict_conversion[input_dict[key]]
        else:
            params.loc[key] = input_dict[key]
        
def create_params_file(simulation):

    """
    Create SPECFEM parameter file from user provided inputs
    """

    write_params(simulation.params_file, simulation.params)

def create_source_file(simulation):

    """
    Create SPECFEM source file from user provided inputs
    """

    write_params(simulation.source_file, simulation.source)

def get_points_towards_ref_station(source_latlon, ref_station, offset_xmin, offset_xmax, N=100):

    """
    Find station coordinates between a source and a reference station
    """

    x = [source_latlon[1], ref_station['lon']]    
    y = [source_latlon[0], ref_station['lat']]
    f = interpolate.interp1d(x, y, fill_value="extrapolate")
    
    xmin = min(source_latlon[1], ref_station['lon'])
    xmin -= kilometer2degrees(offset_xmin/1e3)
    xmax = max(source_latlon[1], ref_station['lon'])
    xmax += kilometer2degrees(offset_xmax/1e3)
    new_lon = np.linspace(xmin, xmax, N)
    np.linspace(source_latlon[1], ref_station['lon'], N)
    new_lat = f(new_lon)
    
    return new_lon, new_lat
    
def get_interface(simulation_domain, input_stations, offset_xmin=10000., offset_xmax=10000., force_topo_at_zero=False):
    
    """
    Create SPECFEM interface input data for parameter and interface files
    """

    ## Simulation domain dimension
    dx = simulation_domain['dx']
    xmin, xmax = -offset_xmin, input_stations.x.max() + offset_xmax
    zmin, zmax = simulation_domain['z-min'], simulation_domain['z-max']
    xx = np.arange(xmin, xmax+dx, dx)
    zz = np.arange(zmin, zmax+dx, dx)
    xmin, xmax = xx.min(), xx.max()
    zmin, zmax = zz.min(), zz.max()
    
    ## Vertical structur
    vertical_points = np.array([zmin, 0., zmax])
    vertical_points_new, ivertical_points_new = [], []
    for z in vertical_points:
        iloc = np.argmin(abs(zz-z))
        ivertical_points_new.append( iloc )
        vertical_points_new.append( zz[iloc] )
    SEM_layers = np.diff(ivertical_points_new)
    layers     = np.diff(vertical_points_new)
    
    if force_topo_at_zero:
        vertical_points_new = [zz-vertical_points_new[1] for zz in vertical_points_new]

    return vertical_points_new, layers, SEM_layers, xmin, xmax, len(xx)

def create_interface_file(simulation):

    """
    Create SPECFEM interface file
    """

    file_format = [
        '# Number of interfaces:',
        '{nb_interfaces}',
        '{interfaces}',
        '# Number of spectral elements in the vertical direction for each layer.',
        '{SEM_interfaces}',
    ]
    
    interface_format = [
        '# Interface {no}',
        '{nb_no}',
        '{points}',
    ]
    
    SEM_interface_format = [
        '# Layer {no}',
        '{nb_SEM}',
    ]
    
    ## Load class attributes
    distance, topo_interp = simulation.distance, simulation.topography
    file_interface = simulation.interface_file
    vertical_points_new = simulation.vertical_points_new
    xmin, xmax = simulation.xmin, simulation.xmax
    SEM_layers, layers = simulation.SEM_layers, simulation.layers
    
    ## Interface points
    list_interface = []
    id_surface = np.argmin(abs(np.array(vertical_points_new)))
    for z in vertical_points_new:
        N_horiz = 2
        loc_dict = {
            'xpoints': np.linspace(xmin, xmax, N_horiz),
            'topo': np.linspace(xmin, xmax, N_horiz)*0. + z,
            'nb_pts': N_horiz,
            'z': z,
            'name': ''
        }
        if z == vertical_points_new[id_surface]:
            loc_dict = {
                'xpoints': distance,
                'topo': topo_interp,
                'nb_pts': len(topo_interp),
                'z': z,
                'name': ' - topography'
            }
        list_interface.append( loc_dict )
    
    ## Create interface text
    interfaces = []
    for iinterface, interface in enumerate(list_interface):
        loc_dict = {
            'no': str(iinterface) + interface['name'],
            'nb_no': interface['nb_pts'],
        }
        for line in interface_format[:-1]:
            interfaces.append( line.format(**loc_dict) )
            
        for x_, z_ in zip(interface['xpoints'], interface['topo']):
            interfaces.append( str(x_) + ' ' + str(z_) )
        
    ## Create layer text
    SEM_interface = []
    for ilayer, _ in enumerate(layers):
        loc_dict = {
            'no': ilayer,
            'nb_SEM': SEM_layers[ilayer],
        }
        for line in SEM_interface_format:
            SEM_interface.append( line.format(**loc_dict) )
    
    loc_dict = {
        'nb_interfaces': len(list_interface),
        'interfaces': '\n'.join(interfaces),
        'SEM_interfaces': '\n'.join(SEM_interface),
    }
    
    all_lines = []
    for line in file_format:
        all_lines.append( line.format(**loc_dict) )
        
    temp_interfaces = file_interface
    file_w = open(temp_interfaces, 'w')
    file_w.write('\n'.join(all_lines))
    file_w.close()
    
    
def add_topography(domain, source_latlon, ref_station, dx, offset_xmin=10000., offset_xmax=10000., 
                   interpolation_method='cubic', low_pass_topography=True,
                   N_discretization_topo=10000, factor_smoothing=10, 
                   add_topography=True, topography_data=None):

    """
    Interpolate interface along source-reference station profile
    """

    if not source_latlon:
        sys.exit('Cannot build topography from ETOPO without lat/lon coordinates for source and stations.')
    
    ## Get interpolation grid
    new_lon, new_lat = get_points_towards_ref_station(source_latlon, ref_station, offset_xmin, offset_xmax, N=N_discretization_topo)

    ## Collect topographic data from ETOPO
    options = {}
    options['region'] = domain.copy()
    options['region']['lat-min'] -= kilometer2degrees(2.*offset_xmin/1e3)
    options['region']['lat-max'] += kilometer2degrees(2.*offset_xmax/1e3)
    options['region']['lon-min'] -= kilometer2degrees(2.*offset_xmin/1e3)
    options['region']['lon-max'] += kilometer2degrees(2.*offset_xmax/1e3)
    if add_topography:
        if topography_data is None:
            topography = collect_topo.collect_region(options)
            lat, lon = np.meshgrid(topography['latitude'], topography['longitude'])
            topo     = topography['topo'].T.ravel()
            points   = np.c_[lon.ravel(), lat.ravel()]
        else:
            points, topo = topography_data[['lon', 'lat']].values, topography_data.topo.values
        
    ## Interpolation
    if not add_topography:
        topo_interp = new_lon*0.
    elif np.diff(new_lat).max() == 0:
        f = interp1d(points[:,0], topo)
        topo_interp = f(new_lon)
    elif np.diff(new_lon).max() == 0:
        f = interp1d(points[:,1], topo)
        topo_interp = f(new_lat)
    else:
        topo_interp = griddata(points, topo, (new_lon, new_lat), method=interpolation_method)
    topo_interp[np.isnan(topo_interp)] = 0. # Remove nan

    distance = []
    gps_ref = gps2dist_azimuth(source_latlon[0], source_latlon[1], 
                               ref_station['lat'], ref_station['lon'])
    
    distance = np.zeros(len(topo_interp))
    for ilonlat, (lon_, lat_) in enumerate(zip(new_lon, new_lat)):
        gps = gps2dist_azimuth(source_latlon[0], source_latlon[1], lat_, lon_)
        
        distance_ = gps[0]
        if abs(gps[2] - gps_ref[2]) > 45.:
            distance_ *= -1
        distance[ilonlat] = distance_
        
    ## Sort arrays
    idx_sort = np.argsort(distance)
    distance = distance[idx_sort]
    topo_interp = topo_interp[idx_sort]
    
    ## Smooth out topography
    if low_pass_topography:
        dx_current = abs(distance[1]-distance[0])
        if dx_current < dx:
            N_new = int(np.ceil(abs(distance[-1]-distance[0]))) / (factor_smoothing * dx)
            Navg  = int(np.ceil(N_discretization_topo/N_new))
            dx_new = 50e3
            Navg = int(dx_new/dx_current)
            topo_interp_orig = topo_interp[:]
            topo_interp = np.convolve(topo_interp_orig, np.ones(Navg)/Navg, mode='valid')
            conv_offset = (distance.size-topo_interp.size)//2
            f = interp1d(distance[conv_offset:-conv_offset-1], topo_interp, bounds_error=False, fill_value='extrapolate')
            topo_interp = f(distance)
        
    return distance, topo_interp
    
def create_seismic_model(simulation, twod_output=True):

    """
    Create SPECFEM-readable seismic model from user provided seismic model
    """
        
    velocity_model = simulation.velocity_model
            
    if twod_output:
        velocity_model_w = np.c_[velocity_model['distance'].values, velocity_model['depth'].values, 
                               velocity_model['rho'].values*1e3, velocity_model['vp'].values*1e3, 
                               velocity_model['vs'].values*1e3, velocity_model['Qp'].values, velocity_model['Qs'].values]
        velocity_model_w = np.append(np.array([[len(velocity_model), 0., 0., 0., 0., 0., 0.]]), velocity_model_w, axis=0)
    
    else:
        velocity_model = velocity_model.groupby('depth').mean().reset_index()
        velocity_model_w = np.c_[velocity_model['depth'].values, 
                               velocity_model['rho'].values*1e3, velocity_model['vp'].values*1e3, 
                               velocity_model['vs'].values*1e3, velocity_model['Qp'].values, velocity_model['Qs'].values]
                               
    np.savetxt(simulation.velocity_file, velocity_model_w)
    
def get_source_latlon(input_source):

    """
    Get source geographical coordinates
    """

    source_latlon = []
    if input_source['coordinates'] == 'latlon':
        source_latlon = [input_source['lat'], 
                         input_source['lon']]
    
    return source_latlon
    
def get_ref_station(stations, ref_station_name):

    """
    Find properties of reference station "ref_station_name"
    """

    ref_station = stations.loc[stations['name'] == ref_station_name, :] 
    if len(ref_station) > 0:
        ref_station = ref_station.iloc[0]
    else:
        sys.exit('Reference station does not exist in station DataFrame')
    
    return ref_station
    
def get_domain(source_latlon, input_stations):

    """
    Get rectangular region that includes the source and the reference station
    """

    domain = {
            'lat-min': 100.,
            'lat-max': -100.,
            'lon-min': 190.,
            'lat-max': -190.,
        }
        
    if source_latlon and 'lat' in input_stations.keys():
        domain['lat-min'] = min(source_latlon[0], input_stations['lat'].min())
        domain['lat-max'] = max(source_latlon[0], input_stations['lat'].max())
        domain['lon-min'] = min(source_latlon[1], input_stations['lon'].min())
        domain['lon-max'] = max(source_latlon[1], input_stations['lon'].max())
        
    return domain
    
def add_domain_to_parfile(xmin, xmax, nx, SEM_layers, params):

    """
    Prepare spatial domain template for SPECFEM input parameter file  
    """
    
    params.loc['xmin'] = xmin
    params.loc['xmax'] = xmax
    params.loc['nx'] = nx
    params.loc['nbregions'] = len(SEM_layers)
    
    ## startingXElement
    format_line = '{i0} {iend} {j0} {jend} {no}'
    lines = []
    jprev = 1
    for ilayer, layer in enumerate(SEM_layers):
        loc_dict = {
            'i0': 1,
            'iend': nx,
            'j0': jprev,
            'jend': layer + jprev - 1,
            'no': ilayer+1
        }
        jprev += layer
        lines.append( format_line.format(**loc_dict) )
    
    params.loc['domain'] = '\n'.join(lines)
    
def get_depths(input_velocity_model):

    """
    Get depth from thickness inputs  
    """
    
    depths = np.cumsum(input_velocity_model.h.values)
    depths = np.concatenate(([0.], depths))
    
    return depths
    
def construct_velocity_model(input_velocity_model):

    """
    Load SPECFEM-readable velocity model with user provided inputs 
    """

    input_velocity_model.reset_index(inplace=True, drop=True)
    depths = get_depths(input_velocity_model)
    
    template_velocity = {
            'distance': 0.,
            'depth': 0.,
            'rho': 0.,
            'vp': 0.,
            'vs': 0.,
            'Qp': 0.,
            'Qs': 0.,
        }
    
    velocity_model = pd.DataFrame()
    for idepth, depth in enumerate(depths[:-1]):
        loc_dict = template_velocity.copy()
        loc_dict['depth'] = depth
        for key in loc_dict:
            if not key in input_velocity_model.keys():
                continue
            loc_dict[key] = input_velocity_model.iloc[idepth].to_dict()[key]
        velocity_model = velocity_model.append( [loc_dict] )
    
    last_depth = velocity_model.iloc[-1].copy()
    last_depth['depth'] = depths[-1]
    velocity_model = velocity_model.append( last_depth )
    velocity_model.reset_index(inplace=True, drop=True)
    
    return velocity_model
    
def load_MSISE(source_latlon, ref_station, zmax, doy, N=1000):

    """
    Build MSIS atmospheric model at the source location projected towards reference station 
    """
        
    if not source_latlon:
        sys.exit('Source lat/lon needed to build MSISE model.')
        
    alts = np.linspace(0., zmax, N)
    lats = source_latlon[0]
    lons = source_latlon[1]
    
    baz_ref_station = gps2dist_azimuth(source_latlon[0], source_latlon[1], ref_station['lat'], ref_station['lon'])[1]
    baz_ref_station = np.radians(baz_ref_station)
    
    model = pd.DataFrame()
    for z in alts:
        model_base = fluids.atmosphere.ATMOSPHERE_NRLMSISE00(z, latitude=lats, longitude=lons, day=doy)
        model_wind = fluids.atmosphere.hwm14(z, latitude=lats, longitude=lons, day=doy)
        projection_wind = model_wind[0]*np.cos(baz_ref_station) + model_wind[1]*np.sin(baz_ref_station)
        rho = model_base.rho
        c   = fluids.atmosphere.ATMOSPHERE_1976.sonic_velocity(model_base.T)
        P   = model_base.P/100.
        T   = model_base.T
        g   = fluids.atmosphere.ATMOSPHERE_1976.gravity(z)
        
        loc_dict = {
            'z': z, 
            't': T,  
            'wx': projection_wind,
            'rho': rho, 
            'p': P*100., 
            'c': c, 
            'g': g
        }
        
        model = model.append( [loc_dict]  )
            
    return model

def resample_custom_model(atmos_model, zmax, N=1000, kind='cubic'):
        
    """
    Interpolate custom model to agree with requested dimensions (maximum altitude and number of points)
    """
        
    if atmos_model.z.max() < zmax:
        sys.exit('Maximum altitude in custom model should be > zmax')
        
    columns_to_interpolate = ['t', 'rho', 'p', 'c', 'g']

    if 'wx' in atmos_model.columns:
        columns_to_interpolate += ['wx']
    else:
        columns_to_interpolate += ['u', 'v']

    if 'cp' in atmos_model.columns:
        columns_to_interpolate += ['cp', 'cv']

    if 'mu' in atmos_model.columns:
        columns_to_interpolate += ['kappa', 'mu']
    if 'taue' in atmos_model.columns:
        columns_to_interpolate += ['taue', 'taus']
        
    ## New altitude vector
    alts = np.linspace(0., zmax, N)
    
    interpolated_atmos_model      = pd.DataFrame(columns=columns_to_interpolate)
    interpolated_atmos_model['z'] = alts
    for column in columns_to_interpolate:
        f = interpolate.interp1d(atmos_model['z'], atmos_model[column], kind=kind)
        interpolated_atmos_model[column] = f(alts)
    
    return interpolated_atmos_model

def load_external_atmos_model(UTC_START, domain, output_dir, use_existing_external_model, dlon = 0.25, dlat = 0.25):

    """
    Load/Download ECMWF external model
    """

    options = {}
    options['UTC_START_list']   = [UTC_START]
    options['REQUEST_REALTIME'] = False
    options['use_specific_atmos_model'] = ''
    options['latbounds'] = [ domain['lat-min'] , domain['lat-max'] ] # min, max latitudes (deg)
    options['lonbounds'] = [ domain['lon-min'] , domain['lon-max'] ] # min, max latitudes (deg)
    options['grid']      = [ dlat , dlon ] # lat, lon steps (deg)
    options['output_dir'] = output_dir
    
    if use_existing_external_model:
        file = output_dir + use_existing_external_model
    elif imported_ECMWF:
        list_files = collect_ECMWF.collect_various_times(options)
        file = list_files[0]
    else:
        sys.exit('You have not imorted the ECWMF libraries')
    return file

def add_params_atmos_model(atmos_model, mu=1e-4, kappa=0., cp=3.5, cv=2.5):

    """
    Add default SPECFEM-required atmospheric parameters if not provided by user
    """
    
    if not 'mu' in atmos_model.columns:
        atmos_model['mu']    = mu
    if not 'kappa' in atmos_model.columns:
        atmos_model['kappa'] = kappa
    if not 'cp' in atmos_model.columns:
        atmos_model['cp'] = cp
    if not 'cv' in atmos_model.columns:
        atmos_model['cv'] = cv
    if not 'gamma' in atmos_model.columns:
        atmos_model['gamma'] = atmos_model['cp']/atmos_model['cv']
    
def create_atmos_model(simulation):

    atmos_model = simulation.atmos_model
    atmos_file  = simulation.atmos_file

    template_atmos = ['z', 'rho', 'dummy0', 'c', 'p', 'dummy1', 'g', 'dummy2', 'kappa', 'mu', 'dummy3', 'dummy4', 'dummy5', 'wx', 'cp', 'cv', 'gamma']
    if 'taue' in atmos_model.columns:
        template_atmos = ['z', 'rho', 'c', 'p', 'g', 'kappa', 'mu', 'wx', 'cp', 'cv', 'gamma', 'taue', 'taus']
    
    for key in template_atmos:
        if not key in atmos_model.keys():
            atmos_model[key] = -1
    
    atmos_model[template_atmos].to_csv(atmos_file, header=None, index=False, sep=' ')
    
def create_instance_mt(input_source):

    ## Create tensor from strike/dip/rake
    mw = input_source['mag']
    m0 = mtm.magnitude_to_moment(mw)  # convert the mag to moment
    
    ## Create Pyrocko moment tensor from vector or strike/dip/rake
    if 'Mnn' in input_source.keys():
        dict_source = {
            'mnn': input_source['Mnn']*m0,
            'mee': input_source['Mee']*m0,
            'mdd': input_source['Mdd']*m0,
            'mne': input_source['Mne']*m0,
            'mnd': input_source['Mnd']*m0,
            'med': input_source['Med']*m0,
        }
        
    else:
        dict_source = {
            'strike': input_source['strike'], 
            'dip': input_source['dip'], 
            'rake': input_source['rake']
        }
    
    mt = mtm.MomentTensor(**dict_source, scalar_moment=m0)
    
    return mt
    
def create_moment_tensor(source, input_source, source_latlon, ref_station, scaling=1e4, mt_coord_system='USE'):

    if not source_latlon and \
        (not 'Mxx' in input_source.keys() or not 'Mxz' in input_source.keys() or not 'Mzz' in input_source.keys()):
        sys.exit('If source lat/lon not provided, Mxx, Mxz and Mzz need to be provided')

    distance = gps2dist_azimuth(source_latlon[0], source_latlon[1], 
                                ref_station['lat'], ref_station['lon'])
    # MODIF 17/10/2022 - why backazimuth selected instead of source-station azimuth?
    #azimuth = distance[2]
    azimuth = distance[1]
    
    ## Create tensor from strike/dip/rake
    mt = create_instance_mt(input_source)
    
    mt = mt.rotated(mtm.rotation_from_angle_and_axis(90-azimuth, [0,0,1])  )
    
    ## Compute SPECFEM moment tensor
    ## Up South East (USE) system
    if mt_coord_system == 'USE':
        mt_use = mt.m6_up_south_east()
        source.loc['Mxx'] = mt_use[2]/scaling
        source.loc['Mxz'] = mt_use[4]/scaling
        source.loc['Mzz'] = mt_use[0]/scaling
    
    ## North East Down (NED) system
    elif mt_coord_system == 'NED':
        mt_use = mt.m6()
        source.loc['Mxx'] = mt_use[1]/scaling
        source.loc['Mxz'] = mt_use[-1]/scaling
        source.loc['Mzz'] = mt_use[2]/scaling
    
    else:
        sys.exit('Coordinate system "{mt_coord_system}" not recognized'.format(mt_coord_system=mt_coord_system))
        
    source.loc['factor'] = 1.
    
    
def update_params_with_topography(source, stations, ref_station_name, distance, topography):

    updated_stations = pd.DataFrame()
    for istation, station in stations.iterrows():
        ix   = np.argmin( abs(np.array(distance) - station['x']) )
        topo = topography[ix]
        station['z'] += topo
        updated_stations = updated_stations.append(station)
    
    ix   = np.argmin( abs(np.array(distance) - source.loc['xs'].iloc[0]) )
    source.loc['zs'] += topography[ix]
    
    ref_station = get_ref_station(updated_stations, ref_station_name)
    
    return updated_stations, ref_station
    
def trim_model_topography(atmos_model, topography, offset=1000.):

    max_topo = topography.max()
    atmos_model.loc[atmos_model['z'] <= max_topo+offset, 'wx'] = 0.
    
def project_uv_wind_model(atmos_model, source_latlon, ref_station):
    
    """
    Project an atmospheric wind model along the source-receiver path
    """
    
    if not 'wx' in atmos_model.columns:
        baz_ref_station = gps2dist_azimuth(source_latlon[0], source_latlon[1], 
                                           ref_station['lat'], ref_station['lon'])[1]
        baz_ref_station = np.radians(baz_ref_station)
        atmos_model['wx'] = atmos_model['v']*np.cos(baz_ref_station) \
                            + atmos_model['u']*np.sin(baz_ref_station)

    #atmos_model = atmos_model[['z', 't', 'wx', 'rho', 'p', 'c', 'g']]
    
    return atmos_model
    
def allowed_stf(type):

    stds = {'Ricker': 1, 'Gaussian_first_derivative': 2, 'Gaussian': 3, 'Dirac': 4, 'Heaviside': 5, 'external': 8, 'burst': 9, 'Gaussian primitive': 10}
    if type in stds.keys():
        return stds[type]
    else:
        return -1
    
"""
SPECFEM simulation preprocessing class 
"""
class create_simulation():

    def __init__(self, specfem_folder='', sample='', template='', source=dict(), station=dict(), ref_station='', simulation_domain=dict(), add_topography=False, topography_data=pd.DataFrame(), force_topo_at_zero=False, interpolation_method_topography='linear', low_pass_topography=True, velocity_model=pd.DataFrame(), parfile=dict(), atmos_model=pd.DataFrame(), ext_mesh=dict(), file_pkl=''):
    
        ## Load previously generated simulation class
        if file_pkl:
            file_to_load = open(file_pkl, 'rb')
            data = pickle.load(file_to_load)
        
        ## Prepare simulation folder
        self.main_folder = specfem_folder
        if not file_pkl:
            self.simu_folder = copy_sample(self.main_folder, sample, template)
        else:
            self.simu_folder = '/'.join(file_pkl.split('/')[:-1])
            
        ## Source preprocessing
        self.input_source = source
        self._update_source()
        
        ## Station preprocessing
        self.input_stations   = station
        self.ref_station_name = ref_station
        self._update_station()
        
        ## Moment tesnor projection
        if 'mt_coord_system' in self.input_source.keys():
            self.mt_coord_system = self.input_source['mt_coord_system']
        else:
            self.mt_coord_system = 'USE'
        self._update_moment_tensor()
        
        self.simulation_domain = simulation_domain
        self.add_topography = add_topography
        self.topography_data = topography_data
        self.domain = get_domain(self.source_latlon, self.input_stations)
        self.offset_xmin = self.simulation_domain['offset_xmin']
        self.offset_xmax = self.simulation_domain['offset_xmax']
        self.force_topo_at_zero = force_topo_at_zero
        self.interpolation_method_topography = interpolation_method_topography
        self.low_pass_topography = low_pass_topography

        ## Prepare data for interface file
        self._update_interface()

        ## Prepare external mesh
        self.file_ext_mesh = ext_mesh['file_ext_mesh']
        self.min_size_element = ext_mesh['min_size_element']
        self.max_size_element_seismic = ext_mesh['max_size_element_seismic']
        self.max_size_element_atmosphere = ext_mesh['max_size_element_atmosphere']
        self._update_external_mesh()
        
        ## Prepare data for velocity file
        self.input_velocity_model = velocity_model
        self._update_velocity()
        
        ## Prepare data for parameter file
        self.input_parfile = parfile
        self._update_parfile()
        
        ## Prepare data for atmospheric file
        self.input_atmos_model = atmos_model
        self._update_atmos()
        
        ## Plot simulation domain if folder wasn't already created
        if not file_pkl:
            file = self.simu_folder + '/simulation_domain.png'
            plot_simulation_domain(self, file)
        
    def save_simulation_parameters(self):
        
        file_to_store = open(self.simu_folder + '/simulation.pkl', 'wb')
        pickle.dump(self, file_to_store)
        
    def _update_parfile(self):
    
        self.params_file = self.simu_folder + '/parfile_input'
        self.params = load_params(self.params_file)
        add_domain_to_parfile(self.xmin, self.xmax, self.nx, self.SEM_layers, self.params)
        update_params(self.params, self.input_parfile)
    
    def _update_source(self):
    
        self.source_file = self.simu_folder + '/source_input'
        self.source = load_params(self.source_file)
        self.source.loc['xs'] = 0.
        self.doy = self.input_source['date'].julday
        self.scaling = self.input_source['scaling']
        self.source_latlon = get_source_latlon(self.input_source)
        self._update_stf()
        
    def _update_stf(self):
    
        no_stf = allowed_stf(self.input_source['stf'])
        if no_stf == -1:
            sys.exit('STF not recognized')
        
        self.source.loc['time_function_type'] = no_stf
        if self.input_source['stf'] == 'external':
            stf_file = self.simu_folder + '/stf.csv'
            self.source.loc['name_of_source_file'] = './stf.csv'
            self.input_source['stf_data'].to_csv(stf_file, header=False, index=False, sep=' ')
            
    def _update_moment_tensor(self):
    
        create_moment_tensor(self.source, self.input_source, self.source_latlon, 
                             self.ref_station, scaling=self.scaling, mt_coord_system=self.mt_coord_system)
        
        update_params(self.source, self.input_source)
        
    def _update_station(self):
    
        self.station_file = self.simu_folder + '/DATA/STATIONS'
        self.stations = load_stations(self.station_file)
        self.input_stations = correct_coordinates_stations(self.input_stations, self.source_latlon)
        self.stations = build_stations(self.stations, self.input_stations)  
        self.ref_station = get_ref_station(self.stations, self.ref_station_name)
        
    def _update_external_mesh(self):

        mesh = dict(
            xtopo = self.distance,
            topo = self.topography, 
            xmin = self.distance.min(), 
            xmax = self.distance.max(), 
            zmin = np.min(self.vertical_points_new), 
            zmax = np.max(self.vertical_points_new)
        )
        opt_gmsh = dict(
            min_size_element = self.min_size_element, 
            max_size_element_seismic = self.max_size_element_seismic, 
            max_size_element_atmosphere = self.max_size_element_atmosphere, 
        )
        _ = create_mesh_gmsh.create_mesh_file_gmsh(self.file_ext_mesh, **mesh, **opt_gmsh)

        bp()

    def _update_interface(self):

        """
        Prepare data for SPEFEM parameter file
        """
        
        self.interface_file = self.simu_folder + '/interfaces_input'
        
        ## If requested, we modify solid-fluid interface to add topography
        opt_topo = dict(
            offset_xmin=self.offset_xmin, 
            offset_xmax=self.offset_xmax, 
            interpolation_method=self.interpolation_method_topography, 
            low_pass_topography=self.low_pass_topography, 
            add_topography=self.add_topography, 
            topography_data=self.topography_data
        )
        self.distance, self.topography = add_topography(self.domain, self.source_latlon, self.ref_station, self.simulation_domain['dx'], **opt_topo)
    
        self.stations, self.ref_station = update_params_with_topography(self.source, self.stations, self.ref_station_name, self.distance, self.topography)
        
        self.vertical_points_new, self.layers, self.SEM_layers, \
            self.xmin, self.xmax, self.nx = get_interface(self.simulation_domain, self.input_stations, 
                                                          offset_xmin=self.offset_xmin, offset_xmax=self.offset_xmax, force_topo_at_zero=self.force_topo_at_zero)
                             
        if not self.add_topography:
            self.topography += self.vertical_points_new[1]
            self.stations, self.ref_station = update_params_with_topography(self.source, self.stations, self.ref_station_name, self.distance, self.topography)
            
    def _update_velocity(self):
    
        self.velocity_file = self.simu_folder + '/velocity_model.txt'
        self.velocity_model = construct_velocity_model(self.input_velocity_model)
        
    def _update_atmos(self):
    
        """
        Construct an atmospheric model either from MSISE or external file
        """
    
        number_altitudes = self.input_atmos_model['number_altitudes']
        self.atmos_file = self.simu_folder + '/atmospheric_model.dat'
        
        ## MSISE atmospheric model
        if not self.input_atmos_model['use_external_atmos_model']:
            self.atmos_model = load_MSISE(self.source_latlon, self.ref_station, 
                                          self.vertical_points_new[-1], self.doy, N=number_altitudes)
        
        ## External atmospheric models
        else:
        
            ## Custom made profiles
            if 'custom_atmospheric_model' in self.input_atmos_model:
                self.atmos_model = self.input_atmos_model['custom_atmospheric_model']
                self.atmos_model = resample_custom_model(self.atmos_model, self.vertical_points_new[-1], 
                                                         N=number_altitudes, kind='cubic')
                
            ## ECMWF profiles
            else:
        
                output_dir = self.input_atmos_model['external_model_directory']
                if output_dir == 'simulation':
                    output_dir = self.simu_folder
                self.dir_atmos_model = load_external_atmos_model(self.input_source['date'], self.domain,
                                                             output_dir, self.input_atmos_model['use_existing_external_model'], 
                                                             dlon = 0.25, dlat = 0.25)
                
                number_points = 5
                source_ = (self.input_source['lat'], self.input_source['lon'])
                receiver = (self.ref_station['lat'], self.ref_station['lon'])
                nc_file  = self.dir_atmos_model
                max_height = self.vertical_points_new[-1]/1e3
                time     = self.input_source['date']
                one_atmos_model = construct_atmospheric_model.atmos_model(source_, receiver, time, max_height, nc_file)
                one_atmos_model.construct_profiles(number_points, number_altitudes, 
                                                   type='specfem', range_dependent=False, 
                                                   projection=True, ref_projection=self.ref_station.to_dict())
                self.atmos_model = one_atmos_model.updated_model
            
        self.atmos_model = project_uv_wind_model(self.atmos_model, self.source_latlon, self.ref_station)
        add_params_atmos_model(self.atmos_model)
        trim_model_topography(self.atmos_model, self.topography)
      
def compute_hlayer_from_depth(seismic_model, unit_depth):
    
    """
    Compute layer thickness from layer depth
    """
    
    new_seismic_model = pd.DataFrame()
    previous_entry = seismic_model.iloc[0].copy()
    for idepth in range(1, len(seismic_model)):
        entry     = seismic_model.iloc[idepth]
        new_entry = previous_entry.copy()
        new_entry['h'] = entry['depth'] - previous_entry['depth']
        if unit_depth == 'km':
            new_entry['h'] *= 1e3
        previous_entry = entry.copy()
        new_seismic_model = new_seismic_model.append( new_entry[['h', 'vp', 'vs', 'rho']] )
        
    return new_seismic_model
      
def load_external_seismic_model(seismic_model_path, add_graves_attenuation=False, columns=['depth', 'vp', 'vs', 'rho'], unit_depth='km', remove_firstlayer=False):
    
    seismic_model = pd.read_csv(seismic_model_path, delim_whitespace=True, header=[0])
    seismic_model.columns = columns
    seismic_model.reset_index(inplace=True, drop=True)
    
    if remove_firstlayer:
        seismic_model = seismic_model.iloc[1:]
        seismic_model.reset_index(inplace=True, drop=True)
    
    if not 'h' in columns:
        seismic_model = compute_hlayer_from_depth(seismic_model, unit_depth)
    elif unit_depth == 'km':
        seismic_model['h'] *= 1e3
    
    if not 'Qs' in seismic_model.keys():
        seismic_model['Qs'] = 9999.
        seismic_model['Qp'] = 9999.

    if add_graves_attenuation:
        seismic_model['Qs'] = 0.05 * seismic_model['vs'] * 1e3
        seismic_model['Qp'] = 2. * seismic_model['Qs'] * 1e3
      
    return seismic_model
      
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):

    import matplotlib.colors as colors

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def add_bball(source, length_domain, max_depth, ax):

    from pyrocko import moment_tensor as mtm
    from obspy.imaging.beachball import beach
    
    offset_x = length_domain/10.
    offset_y = max_depth/5.
    x_orig = source.loc['xs'].iloc[0]/1e3
    y_orig = -1.*source.loc['zs'].iloc[0]/1e3
    x = x_orig + offset_x/1e3
    y = y_orig + offset_y/1e3
    
    input_source = {}
    for key in source.index.tolist(): input_source[key] = source.loc[key].iloc[0]
    mt = create_instance_mt(input_source)
    #m0 = mtm.magnitude_to_moment(source.loc['mag'].iloc[0])  # convert the mag to moment
    #mt = mtm.MomentTensor(strike=source.loc['strike'].iloc[0], dip=source.loc['dip'].iloc[0], 
    #                      rake=source.loc['rake'].iloc[0], scalar_moment=m0)
    
    bball = beach(mt.m6_up_south_east(), xy=(x, y), width=30, linewidth=1, axes=ax, facecolor='tab:blue')
                  
    bball.set_zorder(20)
    ax.add_collection(bball)
    
    ax.plot([x, x_orig], [y, y_orig], color='black', linewidth=3., zorder=5)

def plot_simulation_domain(simulation, file):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cmap_acoustic = plt.get_cmap('plasma')
    cmap_ = plt.get_cmap('gist_earth_r')
    cmap_seismic = truncate_colormap(cmap_, 0.1, 0.8)
    
    source = simulation.source
    stations = simulation.stations
    distance   = simulation.distance
    topography = simulation.topography

    fig = plt.figure(figsize=(10,5))
    grid = fig.add_gridspec(2, 1)
    #fig.set_canvas(plt.gcf().canvas)

    ## Acoustic medium
    axs = []
    axs.append( fig.add_subplot(grid[0, 0]) )
    name_simu = simulation.simu_folder.split('/')[-1]
    axs[0].set_title(name_simu)
    
    atmos   = simulation.atmos_model
    x = np.array([distance.min(), 0., distance.max()])
    x /= 1e3
    z_ = atmos['z'].values[:].reshape(len(atmos), 1)
    z = z_[:]/1e3
    
    wx = atmos['wx'].values.reshape(len(atmos), 1) # assumes range independence
    c  = atmos['c'].values.reshape(len(atmos), 1) # assumes range independence
    color = np.c_[-wx+c, wx+c, wx+c]
    
    sc = axs[0].pcolormesh(x, z, color/1e3, zorder=1, cmap=cmap_acoustic)
    axs[0].set_ylabel('Altitude (km)')
    axs[0].get_xaxis().set_visible(False)
    
    axs[0].fill_between(distance/1e3, topography/1e3, color='black')
    
    axins0 = inset_axes(axs[0], width="2%", height="50%", loc='lower left', bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=axs[0].transAxes, borderpad=0)
    axins0.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar0 = plt.colorbar(sc, cax=axins0, extend='both')
    cbar0.ax.set_ylabel('Effective veloc. (km/s)', rotation=270, labelpad=16)
    
    ## Seismic medium
    axs.append( fig.add_subplot(grid[1, 0], sharex=axs[-1]) )
    max_depth = 100000.
    seismic = simulation.velocity_model
    seismic = seismic.loc[seismic['depth'] < max_depth, :]
    if seismic.depth.max() < atmos['z'].max():
        seismic_bottom = seismic.iloc[-1:].copy()
        seismic_bottom.loc[:, 'depth'] = atmos['z'].max()
        seismic = pd.concat([seismic, seismic_bottom])
    x_ = seismic.distance.unique()
    x  = x_.copy()
    if x_.size == 1:
        x = np.array([distance.min(), distance.max()])
    x /= 1e3
    z = seismic.depth.unique()
    z /= 1e3
    color = seismic['vs'].values.reshape(len(z), len(x_)) # assumes range independence
    if x_.size == 1:
        color = np.c_[color, color]
    sc1 = axs[1].pcolormesh(x, z, color, zorder=1, cmap=cmap_seismic)
    axs[1].invert_yaxis()
    axs[1].set_ylabel('Depth (km)')
    axs[1].set_xlabel('Range (km)')
    ylim = [z.max(), 0.]
    axs[1].set_ylim(ylim)
    
    axins1 = inset_axes(axs[1], width="2%", height="50%", loc='lower left', bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=axs[1].transAxes, borderpad=0)
    axins1.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar1 = plt.colorbar(sc1, cax=axins1, extend='both')
    cbar1.ax.set_ylabel('Shear velocity (km/s)', rotation=270, labelpad=16)
    
    ## Plot source and stations
    length_domain = distance.max() - distance.min()
    
    add_bball(source, length_domain, min(max_depth, seismic['depth'].max()), axs[1])
    axs[1].scatter(source.loc['xs'].iloc[0]/1e3, -1*source.loc['zs'].iloc[0]/1e3, c='yellow', s=200., edgecolors='black', clip_on=False, marker='*', zorder=10)
    for istation, station in stations.iterrows():
        idx_medium = 1
        if station['z'] > 0:
            idx_medium = 0
        axs[idx_medium].scatter(station['x']/1e3, station['z']/1e3, c='tab:orange', s=200., edgecolors='black', clip_on=False, marker='^', zorder=10)
    
    axs[-1].set_xlim(x)

    fig.align_ylabels(axs)
    fig.subplots_adjust(hspace=0., right=0.85)
    
    fig.savefig(file)

def get_gravity_sound_speed_1976(x):

    """
    Compute gravity and adiabatic sound speed from 1976 atmos model specifications
    """
     
    x['c'] = fluids.atmosphere.ATMOSPHERE_1976.sonic_velocity(x['t'])
    x['g'] = fluids.atmosphere.ATMOSPHERE_1976.gravity(x['z'])
    
    return x

def load_Kiruna_profiles_NCPA(file):
    
    """
    Read atmospheric profiles provided by Alexis Le Pichon on 30/04/2021
    """
    
    model = pd.read_csv(file, delim_whitespace=True, header=None)
    model.columns = ['z', 't', 'u', 'v', 'rho', 'p']
    model['z']   *= 1e3
    model['rho'] *= 1e3
    model['p']   *= 1e2
    
    model = model.apply(get_gravity_sound_speed_1976, axis=1)
    
    return model[['z', 't', 'u', 'v', 'rho', 'p', 'c', 'g']]
     
def load_Kiruna_profiles_Alexis(file):
    
    """
    Read atmospheric profiles provided by Alexis Le Pichon on 30/04/2021
    """
    
    model = pd.read_csv(file, delim_whitespace=True, header=None)
    model.columns = ['z', 'u', 'v', 'w', 't', 'rho', 'p']
    model['z']   *= 1e3
    model['rho'] *= 1e3
    model['p']   *= 1e2
    
    model = model.apply(get_gravity_sound_speed_1976, axis=1)
    
    return model[['z', 't', 'u', 'v', 'rho', 'p', 'c', 'g']]

def create_stations_at_given_altitude(source_dict, stations, altitudes, nb_stations=10):

    from pyproj import Geod

    station_template = stations.iloc[0].to_dict()
    
    list_stations = [station_template.copy() for istat in range(nb_stations)]
    stations_update_all = pd.DataFrame()
    for ialt, alt in enumerate(altitudes):

        l_names = [f'ballons_h{no}_alt{ialt}' for no in range(nb_stations)]

        stations_update = pd.DataFrame(list_stations)
        stations_update.loc[:, 'z'] = alt
        stations_update.loc[:, 'name'] = l_names

        source_loc = source_dict['lat'], source_dict['lon']
        stat_ref_loc = stations.iloc[0].lat, stations.iloc[0].lon
        wgs84_geod = Geod(ellps='WGS84')
        l_coord = wgs84_geod.inv_intermediate(source_loc[1], source_loc[0], stat_ref_loc[1], stat_ref_loc[0], nb_stations-1)
        lons = np.array(l_coord.lons).astype(float)
        lons = np.r_[lons, stat_ref_loc[1]]
        lats = np.array(l_coord.lats).astype(float)
        lats = np.r_[lats, stat_ref_loc[0]]
        
        stations_update.loc[:, 'lat'] = lats
        stations_update.loc[:, 'lon'] = lons
        
        stations_update_all = stations_update_all.append(stations_update)

    stations_update_all.reset_index(drop=True, inplace=True)
    return stations_update_all

def create_stations_along_surface(source_dict, stations, nb_stations=10, add_seismic=True, add_array_around_station=True, dx_array=100, nb_in_array=5, add_stations_based_on_angles=[], source_depth=100., nb_stations_based_on_angles=3):

    from pyproj import Geod

    factor_seismic = 1
    if add_seismic:
        factor_seismic = 2
    station_template = stations.iloc[0].to_dict()
    
    list_stations = [station_template.copy() for istat in range(nb_stations*factor_seismic)]
    stations_update = pd.DataFrame(list_stations)
    stations_update.loc[stations_update.index>=nb_stations, 'z'] = -10. # Flag as seismic
    ref_name = stations.iloc[0]['name']
    l_names = ['{ref_name}_arti{no}'.format(ref_name=ref_name, no=no) for no in range(nb_stations-1)] + [ref_name]
    l_names_seismic = ['{name}_seismic'.format(name=name) for name in l_names]
    l_names += l_names_seismic
    stations_update.loc[:, 'name'] = l_names
    
    source_loc = source_dict['lat'], source_dict['lon']
    stat_ref_loc = stations.iloc[0].lat, stations.iloc[0].lon
    wgs84_geod = Geod(ellps='WGS84')
    l_coord = wgs84_geod.inv_intermediate(source_loc[1], source_loc[0], stat_ref_loc[1], stat_ref_loc[0], nb_stations-1)
    
    lons = np.array(l_coord.lons).astype(float)
    lats = np.array(l_coord.lats).astype(float)
    
    stations_update.loc[stations_update.index<nb_stations-1, 'lat'] = lats
    stations_update.loc[stations_update.index<nb_stations-1, 'lon'] = lons
    stations_update.loc[(stations_update.index>=nb_stations)&(stations_update.index<2*nb_stations-1), 'lat'] = lats
    stations_update.loc[(stations_update.index>=nb_stations)&(stations_update.index<2*nb_stations-1), 'lon'] = lons
    
    if add_array_around_station:
        list_stations = [station_template.copy() for istat in range(nb_in_array-1)]
        stations_array = pd.DataFrame(list_stations)
        
        l_names = ['{ref_name}_array{no}'.format(ref_name=ref_name, no=no) for no in range(nb_in_array-1)]
        stations_array.loc[:, 'name'] = l_names
        
        _, az21, _ = wgs84_geod.inv(source_loc[1], source_loc[0], stat_ref_loc[1], stat_ref_loc[0])
        
        dists = np.linspace(dx_array, (nb_in_array-1)*dx_array, nb_in_array-1)
        lons = np.repeat([stat_ref_loc[1]], nb_in_array-1)
        lats = np.repeat([stat_ref_loc[0]], nb_in_array-1)
        azs = np.repeat([az21], nb_in_array-1)
        endlon, endlat, _ = wgs84_geod.fwd(lons, lats, azs, dists)
        stations_array.loc[:, 'lat'] = endlat
        stations_array.loc[:, 'lon'] = endlon
        
        stations_update = stations_update.append( stations_array )
        
    if add_stations_based_on_angles:
    
        list_stations = [station_template.copy() for istat in range(nb_stations_based_on_angles)]
        stations_array = pd.DataFrame(list_stations)
        l_names = ['{ref_name}_angle{no}'.format(ref_name=ref_name, no=no) for no in range(nb_stations_based_on_angles)]
        stations_array.loc[:, 'name'] = l_names
        
        one_range = [np.tan(np.radians(angle))*abs(source_depth) for angle in add_stations_based_on_angles]
        dists = np.linspace(one_range[0], one_range[1], nb_stations_based_on_angles)
        
        az12, _, _ = wgs84_geod.inv(source_loc[1], source_loc[0], stat_ref_loc[1], stat_ref_loc[0])
        
        lons = np.repeat([source_loc[1]], nb_stations_based_on_angles)
        lats = np.repeat([source_loc[0]], nb_stations_based_on_angles)
        azs = np.repeat([az12], nb_stations_based_on_angles)
        endlon, endlat, _ = wgs84_geod.fwd(lons, lats, azs, dists)
        stations_array.loc[:, 'lat'] = endlat
        stations_array.loc[:, 'lon'] = endlon
        
        #print(az12)
        #print(wgs84_geod.inv(lons, lats, endlon, endlat))
        
        stations_update = stations_update.append( stations_array )
        
    stations_update.reset_index(drop=True, inplace=True)
    return stations_update
     
def load_stf(dt, nstep, file='/staff/quentin/Documents/Projects/Kiruna/Celso_data/20200518011156000_crust1se_001_stf.txt', offset_time=0.):

    stf_orig = pd.read_csv(file, delim_whitespace=True, header=None, names=['t', 'amp'])
    f = interpolate.interp1d(stf_orig.t.values, stf_orig.amp.values, fill_value="extrapolate")
    
    
    times = np.arange(0., (nstep+1)*dt, dt)
    amp = f(times)
    amp[times>stf_orig.t.max()] = 0.
    if offset_time > 0:
        times = np.arange(0., (nstep+1)*dt+offset_time, dt)
        add_zeros = 0.*np.arange(0., offset_time, dt)
        amp = np.r_[add_zeros, amp]
        
    stf = pd.DataFrame()
    stf['t'] = times
    stf['amp'] = amp
    
    return stf