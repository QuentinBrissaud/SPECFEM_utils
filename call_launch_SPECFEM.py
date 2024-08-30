#!/usr/bin/env python3

## Standard libraries
from pdb import set_trace as bp
import pandas as pd
import numpy as np
import sys
from obspy.core.utcdatetime import UTCDateTime
from pyrocko import moment_tensor as pmt

## Custom library
sys.path.append("./SPECFEM_utils/")
import launch_SPECFEM as LS

##########################
if __name__ == '__main__':
    
    ## 
    options = {}
    options['specfem_folder'] = './'
    options['sample']         = f'{options["specfem_folder"]}example_folder/'
    #options['template']       = 'simulation_flores_{no}'
    #options['template']       = 'simulation_alaska_{no}'
    options['template']       = 'simulation_venus_{no}'
    
    #####################
    ## Atmospheric models
    atmos_file = './data_venus/atmospheric_model_updated.dat'
    options['atmos_model'] = {
        'use_external_atmos_model': True,
        'type_external_atmos_model': 'custom',
        'number_altitudes': 1000,
        'custom_atmospheric_model': LS.load_SPECFEM_profile(atmos_file),
    }

    #############################
    ## SPECFEM default parameters 
    options['parfile'] = {
        'title': 'Venusquake simulation',
        'USE_DISCONTINUOUS_METHOD': True,
        'ATTENUATION_VISCOELASTIC_SOLID': True,
        'USE_LNS': True, # Only for new SPECFEM version
        'REMOVE_DG_FLUID_TO_SOLID': False,
        'CONSTRAIN_HYDROSTATIC': True,
        'DEACTIVATE_SEISMIC_AFTER_T': True,
        'timeval_max_elastic': 400.,
        'velocity_mesh': 340.,
        'size_active_mesh': 100.,
        'x0_init_active_mesh': 0.,
        'activate_moving_mesh': False,
        'NPROC': 200,
        'NSTEP': 1000000,
        'DT': 2.5e-3,
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
        'read_external_mesh': True,
        }
    
    ############################
    ## Simulation domain default
    options['simulation_domain'] = {
        'z-min': -50000.,
        'z-max': 70000.,
        #'dx': 350.,
        'dx': 500.,
        'offset_x': 10000.
        }
    options['simulation_domain'].update({'offset_xmin': options['simulation_domain']['dx']*20,'offset_xmax': options['simulation_domain']['dx']*20,})

    ##################
    ## Station default
    stations_default = pd.DataFrame()
    
    options['ref_station']  = 'SANI'
    station = {
        'lat': -2.05,
        'lon': 125.99,
        'z': 10., # above topography
        'name': 'SANI',
        'array': 'GE',
        'coordinates': 'latlon'
        }
    
    options['ref_station']  = 'Q23K'
    station = {
        'lat': 59.429600,
        'lon': -146.339900,
        'z': 10., # above topography
        'name': 'Q23K',
        'array': 'AK',
        'coordinates': 'latlon'
        }
    
    options['ref_station']  = 'P16k'
    station = {
        'lat': 59.0314,
        'lon': -157.9906,
        'z': 10., # above topography
        'name': 'P16k',
        'array': 'AK',
        'coordinates': 'latlon'
        }
    
    options['ref_station']  = 'balloon'
    station = {
        'lat': -45., 
        'lon': 0.,
        'z': 50.e3, # above topography
        'name': 'balloon',
        'array': 'VE',
        'coordinates': 'latlon'
        }

    add_stations_based_on_angles = []
    nb_stations_based_on_angles = 3
    stations_default = pd.concat([stations_default, pd.DataFrame([station])])

    ####################
    ## Source parameters
    # Source default
    source_default = {
        'coordinates': 'latlon',
        'lat': -7.603, # dummy
         'lon': 122.227, # dummy
        'zs': -10000., # dummy
        'date': UTCDateTime('2021-12-14T03:20:23'), # dummy
        'scaling': 1e8,
        'strike': 290, # dummy
        'dip': 89., # dummy
        'rake': 177., # dummy
        'mag': 7.3, 
        'stf_data': 'Gaussian', # dummy
        'mt_coord_system': 'USE'
        }

    stfs = ['Dirac', ]

    # Flores
    mechanisms = []
    f0s = [1./12.11]
    strike = 190.
    dip = 89.
    rake = 177.
    m6 = pmt.MomentTensor(strike=strike, dip=dip, rake=rake).m6()
    mechanisms.append({'Mnn': m6[0], 'Mne': m6[3], 'Mnd': m6[4], 'Mee': m6[1], 'Med': m6[5], 'Mdd': m6[2],  }) 
    location_source = dict(lat= -7.603, lon= 122.227, date=UTCDateTime('2021-12-14T03:20:23')) # Flores
    source_default.update(location_source)
    depths = [-17500.]

    # Alaska
    mechanisms = []
    f0s = [1./30.31]
    strike = 239.
    dip = 14.
    rake = 95.
    m6 = pmt.MomentTensor(strike=strike, dip=dip, rake=rake).m6()
    mechanisms.append({'Mnn': m6[0], 'Mne': m6[3], 'Mnd': m6[4], 'Mee': m6[1], 'Med': m6[5], 'Mdd': m6[2],  })
    location_source = dict(lat= 55.3635, lon= -157.8876, date=UTCDateTime('2021-07-29T06:15:49.188000Z')) # Alaska 8.2 https://earthquake.usgs.gov/earthquakes/eventpage/ak0219neiszm/executive
    source_default.update(location_source)
    depths = [-35500.]

    # Venus
    mechanisms = []
    f0s = [1./30.31]
    mechanisms.append({'Mnn': 0., 'Mne': 0., 'Mnd': -0., 'Mee': -0., 'Med': 1., 'Mdd': 1.,  }) # Normal fault
    location_source = dict(lat= -90., lon= 0., date=UTCDateTime('2021-07-29T06:15:49.188000Z')) # Alaska 8.2 https://earthquake.usgs.gov/earthquakes/eventpage/ak0219neiszm/executive
    source_default.update(location_source)
    depths = [-35500.]

    #############
    ## Topography
    options['add_topography'] = True

    file = './data_venus/quickmap-profile-data_profile_topo_large.csv'
    options['topography_data'] = LS.read_venus_topography(file, source_default['lat'], source_default['lon'], station['lat'], station['lon'], options['simulation_domain']['offset_x'], R0=6052000)
    options['fit_vel_to_topo'] = False
    #options['topography_data'] = None
    #options['fit_vel_to_topo'] = True
    #import matplotlib.pyplot as plt; plt.figure(); plt.plot(options['topography_data'].R, options['topography_data'].topo); plt.savefig('./test_topo.png')
   
    options['force_topo_at_zero'] = True
    options['interpolation_method_topography'] = 'linear'
    options['low_pass_topography'] = False

    ################
    ## External mesh
    options['ext_mesh'] = dict(
        min_size_element = options['simulation_domain']['dx'], 
        #max_size_element_seismic = 2500.,  
        #max_size_element_seismic = 1500., 
        max_size_element_seismic = 2500., 
        max_size_element_atmosphere = options['simulation_domain']['dx'], 
        use_pygmsh = True,
        use_cpml = True,
        save_mesh = True,
        factor_transition_zone = 10.,
        factor_pml_lc_g = 10.,
        alpha_taper = 0.1,
    )
    options['parfile']['read_external_mesh'] = False
    
    ###########################
    ## Seismic model parameters
    remove_firstlayer=False
    seismic_models, dir_models = {}, []
    seismic_model_path = './data_flores/model_seismic_flores_2d.csv'
    seismic_model_path = './data_flores/model_seismic_flores_2d_notopo.csv'
    seismic_model_path = '.\data_alaska\model_seismic_alaska8.2_2d_notopo.csv'
    seismic_model_path = '.\data_alaska\model_seismic_alaska8.2_2d_notopo_P16K.csv'
    twod_output=True
    #dir_models.append(seismic_model_path)
    #seismic_models.update( {seismic_model_path: LS.load_external_seismic_model(seismic_model_path, add_graves_attenuation=False, columns=['distance', 'h', 'depth', 'vs', 'vp', 'rho', 'Qs', 'Qp'], unit_depth='km', remove_firstlayer=remove_firstlayer)} )

    seismic_model_path='./data_venus/quickmap-profile-data_crust_topo_large.csv'; 
    dir_models.append(seismic_model_path)
    model = LS.build_venus_model(seismic_model_path, vs_crust=3.5, vp_crust=6., rho_crust=2.8, Qp_crust=1500., Qs_crust=600., vs_mantle=4.4, vp_mantle=7.5, rho_mantle=3.3, Qp_mantle=1500., Qs_mantle=600., h_mantle=1000.)
    seismic_models.update( {seismic_model_path: model} )

    #########################
    ## Create parameter space
    parameters = LS.create_parameter_space(mechanisms, f0s, stfs, dir_models, depths)

    # from importlib import reload;
    # reload(LS)
    # LS.create_stations_at_given_altitude(source_default, stations_default, [5000.], nb_stations=10)

    ###################################################################
    ## Loop over requested parameter space to create simulation folders
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
        options['station'] = LS.create_stations_along_surface(options['source'], options['station'], nb_stations=10, add_seismic=True, add_array_around_station=False, dx_array=100, nb_in_array=5, add_stations_based_on_angles=add_stations_based_on_angles, source_depth=depth, nb_stations_based_on_angles=nb_stations_based_on_angles)
        alts = [10e3, 20e3, 30e3]
        nb_stations = 10
        stations = options['station'].loc[options['station']['name']==options['ref_station']]
        options['station'] = pd.concat([options['station'], LS.create_stations_at_given_altitude(options['source'], stations, alts, nb_stations=nb_stations)])
        options['station'].reset_index(drop=True, inplace=True)
        
        ## Create simulation
        simulation = LS.create_simulation(**options, file_pkl='')
        LS.create_params_file(simulation)
        LS.create_source_file(simulation)
        LS.create_station_file(simulation)
        LS.create_interface_file(simulation)
        LS.create_seismic_model(simulation, twod_output=twod_output)
        LS.create_atmos_model(simulation)
        simulation.save_simulation_parameters()
        
        ## Parameter to print
        parameter_file = simulation.simu_folder + '/parameters.csv'
        mechanism.update({'no': simulation.simu_folder.split('_')[-1], 'dir_simu': simulation.simu_folder, 'f0': f0, 'stf': stf, 'atmos_file': 'ncpa', 'seismic_model': dir_model, 'zs': depth})
        parameter_txt = pd.DataFrame([mechanism])
        parameter_txt.to_csv(parameter_file, header=True, index=False)
        all_parameters_txt = pd.concat([all_parameters_txt, parameter_txt])

        ## Plot domain
        from importlib import reload;  reload(LS);  file = f'{simulation.simu_folder}/simulation_domain.png'; LS.plot_simulation_domain(simulation, file, n_depths=100, max_depth=abs(options['simulation_domain']['z-min'])/1e3)
        
    #LS.create_all_parameters(options['specfem_folder'], all_parameters_txt, options['template'])
    #import matplotlib.pyplot as plt; plt.figure(); plt.plot(simulation.distance, simulation.topography); plt.savefig('./test_topo.png')
    bp()