#!/usr/bin/env python3

## Standard libraries
from pdb import set_trace as bp
import os 
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from obspy.core.utcdatetime import UTCDateTime

## Custom library
import launch_SPECFEM as LS

def create_sentence_from_dict(name, one_dict):
    dict_inside = ''
    one_key = '{key}: {value}, '
    for key in one_dict: 
        dict_inside += one_key.format(key=key, value=one_dict[key])
    
    return dict_inside
        
def create_text_from_parameters(mechanism, f0, stf, seismic_model, depth):
    mechanism_txt = create_sentence_from_dict('mechanism', mechanism)
    f0_txt = create_sentence_from_dict('f0', f0)
    stf_txt = create_sentence_from_dict('stf', stf)
    depth_txt = create_sentence_from_dict('depth', depth)
    all_txt = [mechanism_txt, f0_txt, stf_txt, depth_txt, seismic_model]
    all_txt = '\n'.join(all_txt)
    
    return all_txt

def create_dataframe_from_parameters(mechanism, f0, stf, seismic_model, depth):

    entry = {'seismic_model': seismic_model}
    entry.update(mechanism)
    entry.update(f0)
    entry.update(stf)
    entry.update(depth)
    entry = pd.DataFrame([entry])
    return entry

def create_all_parameters(output_dir, all_parameters, template):

    name = template.format(no=f'{all_parameters.no.min()}-{all_parameters.no.max()}')
    all_parameters.reset_index(drop=True, inplace=True)
    parameter_file = '{output_dir}/parameters_{name}.csv'.format(output_dir=output_dir, name=name)
    all_parameters.to_csv(parameter_file, header=True, index=True)
    print('Simulations generated:')
    print(all_parameters)

## Parameter space
def create_parameter_space(mechanisms, f0s, stfs, dir_models, depths):
    import itertools
    parameter_space = []
    parameter_space.append(mechanisms)
    parameter_space.append(f0s)
    parameter_space.append(stfs)
    parameter_space.append(dir_models)
    parameter_space.append(depths)
    parameters = list(itertools.product(*parameter_space))
    
    return parameters

def load_SPECFEM_profile(file, deactivate_attenuation=False):
    
    """
    Read atmospheric profiles used in SPECFEM
    """
    
    model = pd.read_csv(file, delim_whitespace=True, header=[0])
    model.columns = ['z','rho','t','c','p','n/a','g','n/a.1','kappa','mu','n/a.2','v','u','w_proj[m/s]','cp','cv','gamma[1]','fr[Hz]','Svib[1]','kappa1','taus','taue','taus_new','taue_new']
    if deactivate_attenuation:
        model['kappa'] = 0.
        model['mu'] = 0.
        model['taue'] = 1.
        model['taus'] = 1.

    return model[['z', 't', 'u', 'v', 'rho', 'p', 'c', 'g', 'kappa', 'mu', 'cp', 'cv', 'taue', 'taus']]

##########################
if __name__ == '__main__':
    
    ## 
    options = {}
    options['specfem_folder'] = '../../EXAMPLES/'
    options['sample']         = f'{options["specfem_folder"]}example_folder/'
    options['template']       = 'simulation_test_Celine_{no}'
    
    ## Topography
    options['add_topography'] = False
    options['force_topo_at_zero'] = True
    
    ## Atmospheric models
    atmos_file = './test_data_Venus/atmospheric_model_updated.dat'
    options['atmos_model'] = {
        'use_external_atmos_model': True,
        'number_altitudes': 1000,
        'custom_atmospheric_model': load_SPECFEM_profile(atmos_file, deactivate_attenuation=False),
    }
    
    ## SPECFEM default parameters 
    options['parfile'] = {
        'title': 'Venusquake simulation',
        'USE_DISCONTINUOUS_METHOD': True,
        'ATTENUATION_VISCOELASTIC_SOLID': True,
        'USE_LNS': True, # Only for new SPECFEM version
        'REMOVE_DG_FLUID_TO_SOLID': False,
        'CONSTRAIN_HYDROSTATIC': True,
        'DEACTIVATE_SEISMIC_AFTER_T': True,
        'timeval_max_elastic': 40.,
        'velocity_mesh': 340.,
        'size_active_mesh': 100.,
        'x0_init_active_mesh': 0.,
        'activate_moving_mesh': False,
        'NPROC': 60,
        'NSTEP': 100000,
        'DT': 1.5e-3,
        'ABC_STRETCH_TOP': True,                # Use buffer-based stretching method on that boundary?
        'ABC_STRETCH_LEFT': True,                # Use buffer-based stretching method on that boundary?
        'ABC_STRETCH_BOTTOM': False,                # Use buffer-based stretching method on that boundary?
        'ABC_STRETCH_RIGHT': True,                # Use buffer-based stretching method on that boundary?
        'ABC_STRETCH_TOP_LBUF': 10.,                  # Length of the buffer used for the buffer-based stretching method.
        'ABC_STRETCH_LEFT_LBUF': 10.,                    # Length of the buffer used for the buffer-based stretching method.
        'ABC_STRETCH_BOTTOM_LBUF': 10.,                    # Length of the buffer used for the buffer-based stretching method.
        'ABC_STRETCH_RIGHT_LBUF': 10.,                    # Length of the buffer used for the buffer-based stretching method.
        'NSTEP_BETWEEN_OUTPUT_IMAGES': 500,
        'output_wavefield_dumps': False,
        'NSTEP_BETWEEN_OUTPUT_WAVE_DUMPS': 5000,
        'factor_subsample_image': 1.,
        'STACEY_ABSORBING_CONDITIONS': False,
        'PML_BOUNDARY_CONDITIONS': True,
        'NELEM_PML_THICKNESS': 10,
        'time_stepping_scheme': 2,
        # Available models.
        #   default:     define model using nbmodels below.
        #   ascii:       read model from ascii database file.
        #   binary:      read model from binary database file.
        #   external:    define model using 'define_external_model' subroutine.
        #   legacy:      read model from 'model_velocity.dat_input'.
        #   external_DG: read DG model from 'atmospheric_model.dat' (generated with 'utils_new/Atmospheric_Models/Earth/wrapper/msisehwm'), and other models with "Velocity and Density Models" below.
        'MODEL': 'external_DG',
        }
    
    ## Simulation domain default
    options['simulation_domain'] = {
        'z-min': -25000.,
        'z-max': 9000.,
        'dx': 1000.,
        'offset_x': 10000.
        }
    options['simulation_domain'].update({'offset_xmin': options['simulation_domain']['dx']*20,'offset_xmax': options['simulation_domain']['dx']*20,})
    
    ## Station default
    stations_default = pd.DataFrame()
    
    options['ref_station']  = 'dummy_station'
    station = {
        'lat': 0.,
        'lon': 1,
        'z': 10., # above topography
        'name': 'dummy_station',
        'array': 'NO',
        'coordinates': 'latlon'
        }
    add_stations_based_on_angles = []
    nb_stations_based_on_angles = 3
    stations_default = stations_default.append( [station] )
    
    ## Source default
    source_default = {
        'coordinates': 'latlon',
        'lat': 0.,
        'lon': 0.,
        'zs': -10000.,
        'date': UTCDateTime(2020, 5, 18, 1, 11, 57),
        'scaling': 1e8,
        'strike': 348.,
        'dip': 35.,
        'rake': 50.,
        'mag': 4.6,
        'stf_data': LS.load_stf(options['parfile']['DT'], options['parfile']['NSTEP'], file='/staff/quentin/Documents/Projects/Kiruna/Celso_data/20200518011156000_crust1se_001_stf.txt'),
        'mt_coord_system': 'USE'
        }
    
    remove_firstlayer=False
    
    ## Source parameters
    f0s = [0.3]
    mechanisms = []
    mechanisms.append({'Mnn': 0., 'Mne': 0., 'Mnd': -0., 'Mee': -0., 'Med': 1., 'Mdd': 1.,  }) # Normal fault
    stfs = ['Gaussian', ]
    
    ## Model parameters
    seismic_models, dir_models = {}, []
    seismic_model_path = './test_data_Venus/venuscrust.txt'
    dir_models.append(seismic_model_path)
    seismic_models.update( {seismic_model_path: LS.load_external_seismic_model(seismic_model_path, add_graves_attenuation=False, columns=['h', 'vs', 'vp', 'rho', 'Qs', 'Qp'], unit_depth='km', remove_firstlayer=remove_firstlayer)} )
    depths = [-10000.]
    
    parameters = create_parameter_space(mechanisms, f0s, stfs, dir_models, depths)

    # from importlib import reload;
    # reload(LS)
    # LS.create_stations_at_given_altitude(source_default, stations_default, [5000.], nb_stations=10)
    
    ## Loop over requested parameter space
    all_parameters_txt = pd.DataFrame()
    for mechanism, f0, stf, dir_model, depth in parameters:
        
        ## Create source
        options['source'] = source_default.copy()
        options['source'].update(mechanism)
        options['source'].update({'f0': f0})
        options['source'].update({'stf': stf})
        options['source'].update({'zs': depth})
       
        ## Velocity model
        options['velocity_model'] = seismic_models[dir_model]
        
        ## Stations
        options['station'] = stations_default.copy()
        options['station'] = LS.create_stations_along_surface(options['source'], options['station'], nb_stations=10, add_seismic=True, add_array_around_station=True, dx_array=100, nb_in_array=5, add_stations_based_on_angles=add_stations_based_on_angles, source_depth=depth, nb_stations_based_on_angles=nb_stations_based_on_angles)
        alt = 2000.
        nb_stations = 10
        stations = options['station'].loc[options['station']['name']==options['ref_station']]
        options['station'] = options['station'].append( LS.create_stations_at_given_altitude(options['source'], stations, [alt], nb_stations=nb_stations) )
        options['station'].reset_index(drop=True, inplace=True)
        
        ## Compute right spatial step - not used
        #v_acous = 0.34
        #dx = 0.5 * v_acous*1e3/f0
        
        ## Create simulation
        simulation = LS.create_simulation(options, file_pkl='')
        LS.create_params_file(simulation)
        LS.create_source_file(simulation)
        LS.create_station_file(simulation)
        LS.create_interface_file(simulation)
        LS.create_velocity_model(simulation, twod_output=False)
        LS.create_atmos_model(simulation)
        simulation.save_simulation_parameters()
        
        ## Parameter to print
        parameter_file = simulation.simu_folder + '/parameters.csv'
        #parameter_txt = create_dataframe_from_parameters(mechanism, {'f0': f0}, {'stf': stf}, dir_model, {'zs': depth})
        mechanism.update({'no': simulation.simu_folder.split('_')[-1], 'dir_simu': simulation.simu_folder, 'f0': f0, 'stf': stf, 'atmos_file': atmos_file, 'seismic_model': dir_model, 'zs': depth})
        parameter_txt = pd.DataFrame([mechanism])
        parameter_txt.to_csv(parameter_file, header=True, index=False)
        all_parameters_txt = all_parameters_txt.append(parameter_txt)
        
    create_all_parameters(options['specfem_folder'], all_parameters_txt, options['template'])
    bp()