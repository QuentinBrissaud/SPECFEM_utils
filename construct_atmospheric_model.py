#!/usr/bin/env python3
from netCDF4 import Dataset
import numpy as np
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import pandas as pd
import fluids
from scipy import interpolate
import os
from scipy.interpolate import griddata
from obspy.core.utcdatetime import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth    
import great_circle_calculator.great_circle_calculator as gcc

try:
    import plotting_routines
except:
    print('Plotting routines not available')
    
def build_locations(source, receiver, number_points):
    
    fractions = np.linspace(0., 1., number_points)
    
    list_of_points = []
    for fraction in fractions:
        midpoint = gcc.intermediate_point(source[::-1], receiver[::-1], fraction=fraction)
        distance = gcc.distance_between_points(source[::-1], midpoint, unit='meters', haversine=True)
        list_of_points.append( list(midpoint) + [distance/1e3] )
        
    list_of_points = np.array(list_of_points)
   
    return list_of_points
   
def correct_latlon_outside_range(list_of_points, points, default_offset_interp=1e-5):

    """
    Correct latitude and longitude of interpolation points if outside of NETCDF model range
    """

    new_lon, new_lat = list_of_points[:, 0], list_of_points[:, 1]
    new_lon, new_lat = np.round(new_lon, 5), np.round(new_lat, 5)
    maxlat = points[:,1].max()
    maxlon = points[:,0].max()
    minlat = points[:,1].min()
    minlon = points[:,0].min()
    new_lon[new_lon>=maxlon] -= default_offset_interp
    new_lat[new_lat>=maxlat] -= default_offset_interp
    new_lon[new_lon<=minlon] += default_offset_interp
    new_lat[new_lat<=minlat] += default_offset_interp
    
    return new_lon, new_lat
   
def is_point_outside_domain(points, new_point):

    """
    Check if a location in "new_point" is outside the rectangular boundaries defined by points
    """

    outside = False
    if np.array(new_point[0]).min() < points[:,0].min() \
        or np.array(new_point[0]).max() > points[:,0].max() \
        or np.array(new_point[1]).min() < points[:,1].min() \
        or np.array(new_point[1]).max() > points[:,1].max():
        outside = True
   
    return outside
   
def interpolate_external_model(list_of_points_in, lats_in, lons_in, heights, 
                               levels, zonal_in, meridonial_in, temperature_in, kind_1d_interp='cubic', 
                               default_offset_interp=1e-5, offset=7., handle_ensembles=-1):
    
    list_of_points = list_of_points_in.copy()
    lons, lats = lons_in.copy(), lats_in.copy()
    
    print('Build grid')
    
    ## Added on 03.02.2022 to reduce computational time -> just interpolate in the vicinity of requested points
    min_lat, max_lat = list_of_points[:,1].min()-offset, list_of_points[:,1].max()+offset
    min_lon, max_lon = list_of_points[:,0].min()-offset, list_of_points[:,0].max()+offset
    
    flag=False
    if min_lon < lons.min() or max_lon > lons.max():
        lons[lons<0] += 360.
        list_of_points[:,0][list_of_points[:,0] < 0] += 360.
        flag=True
        
    if min_lat < lats.min() or max_lat > lats.max():
        lats[lats<0] += 180.
        list_of_points[:,1][list_of_points[:,1] < 0] += 180.
        flag=True
        
    min_lat, max_lat = list_of_points[:,1].min()-offset, list_of_points[:,1].max()+offset
    min_lon, max_lon = list_of_points[:,0].min()-offset, list_of_points[:,0].max()+offset
    LAT, LON = np.meshgrid(lats.filled(), lons.filled())
    
    idx_lat = np.where((lats.filled() <= max_lat) & (lats.filled() >= min_lat))[0]
    if idx_lat.size == 0:
        idx_lat = np.array([np.argmin(abs(lats.filled()-max_lat))])
    idx_lon = np.where((lons.filled() <= max_lon) & (lons.filled() >= min_lon))[0]
    if idx_lon.size == 0:
        idx_lon = np.array([np.argmin(abs(lons.filled()-max_lon))])
    
    LAT = LAT[:, idx_lat]
    LAT = LAT[idx_lon, :]
    LON = LON[:, idx_lat]
    LON = LON[idx_lon, :]
    
    print('Start interpolating')
    
    ## If ensembles, take average
    zonal = zonal_in[:]
    meridonial = meridonial_in[:]
    temperature = temperature_in[:]
    if len(zonal.shape) == 5:
        if handle_ensembles == -1:
            zonal = np.mean(zonal, axis=1)
            meridonial = np.mean(meridonial, axis=1)
            temperature = np.mean(temperature, axis=1)
        else:
            zonal = zonal[:,handle_ensembles,:,:,:]
            meridonial = meridonial[:,handle_ensembles,:,:,:]
            temperature = temperature[:,handle_ensembles,:,:,:]
    
    ## Select specific regions to save computational time
    zonal = zonal[:,:,idx_lat, :]
    zonal = zonal[:,:,:, idx_lon]
    meridonial = meridonial[:,:,idx_lat, :]
    meridonial = meridonial[:,:,:, idx_lon]
    temperature = temperature[:,:,idx_lat, :]
    temperature = temperature[:,:,:, idx_lon]
    points   = np.c_[LON.ravel(), LAT.ravel()]
    variables = {'u': zonal, 'v': meridonial, 't': temperature}
    
    new_lon, new_lat = correct_latlon_outside_range(list_of_points, points, \
                            default_offset_interp=default_offset_interp)
    
    print('Interpolate variables')
    
    atmos_model = {}
    for variable in variables:
        
        atmos_model[variable] = []
        for iheight, height in enumerate(heights):
            
            field = variables[variable]
            """
            if len(field.shape) == 5:
                field = field[0, :, iheight, :, :]
                field = np.mean(field, axis=0)
            else:
                field = field[0, iheight, :, :]
            """
            field = field[0, iheight, :, :]
            field = field.ravel()
            
            if field.size <= 3:
                field_interp = np.zeros(new_lon.shape) + np.mean(field)

            ## Only one longitude available
            elif np.unique(points[:,0]).size == 1:
                f = interpolate.interp1d(points[:,1], field, kind=kind_1d_interp, fill_value='extrapolate')
                field_interp = f(new_lat)

            ## Only one latitude available
            elif np.unique(points[:,1]).size == 1:
                f = interpolate.interp1d(points[:,0], field, kind=kind_1d_interp, fill_value='extrapolate')
                field_interp = f(new_lon)

            else:
                ## Below makes sure that for points outside of region boundaries, we select the closest point within the boundaries
                ## It assumes new_lat.size = 1
                #if new_lat < points[:,1].min() or new_lat > points[:,1].max()
                #    new_lat[0] = points[np.argmin(abs(new_lat[0]-points[:,1])),1]
                #if new_lon < points[:,1].min() or new_lon > points[:,1].max()
                #    new_lon[0] = points[np.argmin(abs(new_lon[0]-points[:,1])),0]
                try:
                 if not is_point_outside_domain(points, (new_lon, new_lat)):
                    field_interp = griddata(points, field, (new_lon, new_lat), method='cubic')
                
                 else:
                    field_interp = griddata(points, field, (new_lon, new_lat), method='nearest')
                except:
                 bp()
            if field_interp[np.isnan(field_interp)].size > 0:
                bp()
                
            field_interp[np.isnan(field_interp)] = 0. # Remove nan
    
            atmos_model[variable].append( field_interp )
        
        atmos_model[variable] = np.array( atmos_model[variable] )

    return atmos_model
   
def plot_profiles(dir, model):

    fig, axs = plt.subplots(1, 1)
    
    heights = model.z.unique()
    list_of_points = model['distance'].unique()
    
    colors = ['tab:blue', 'tab:purple']
    variables = ['u', 'v']
    for ivariable, variable in enumerate(variables):
        
        for i, distance in enumerate(list_of_points):
            label = {}
            if i == 0:
                label = {'label': variable}
                
            field = model.loc[model['distance'] == distance, variable].values
            field = field/abs(model[variable].values).max()
                
            axs.plot(field + i*2, np.array(heights)/1e3, color=colors[ivariable], zorder=10, **label)
            
    axs.scatter([0.], [0.], c='yellow', s=200., edgecolors='black', clip_on=False, marker='*', zorder=100)
    axs.scatter([2*(len(list_of_points)-1)], [0.], c='orange', s=200., edgecolors='black', clip_on=False, marker='^', zorder=100)
            
    labels = ['0.0'] + [str(round(distance,1)) for distance in list_of_points]
    axs.set_xticklabels(labels)
    axs.set_ylim([heights[-1]/1e3, heights[0]/1e3])
    axs.set_xlabel('Range (km)')
    axs.set_ylabel('Altitude (km)')
    axs.grid(zorder=10)
    axs.legend()
    
    plt.savefig(dir + 'profiles.pdf')
   
import spaceweather as sw
def run_MSISHWM_wrapper(location, date_UTC, zmax, N, file_sw='/projects/active/infrasound/data/infrasound/2021_seed_infrAI/model_atmos_fixed/spaceweather.csv', bin_dir='/staff/quentin/Documents/Codes/msis20hwm14/'):
    
    ## Collect space weather parameters 
    if file_sw:
        df_MSIS = pd.read_csv(file_sw, header=[0], parse_dates=['index'])
    else:
        df_MSIS = sw.sw_daily()
        df_MSIS.reset_index(inplace=True)
        #df_MSIS.to_csv(file_sw, header=True, index=False)
        
    df_MSIS = df_MSIS.iloc[abs(df_MSIS['index']-date_UTC.datetime).idxmin()]
    
    ## Parameters
    sec_UTC = date_UTC - UTCDateTime(date_UTC.year, date_UTC.month, date_UTC.day)
    F107  = df_MSIS.f107_adj
    F107A = df_MSIS.f107_81ctr_adj
    AP = df_MSIS.Apavg
    
    ## Run wrapper
    output = 'msisehwm_model_output_{pid}'.format(pid=os.getpid())
    cmd_format = './msis {min_alt} {max_alt} {N_samples} {lat} {lon} {year} {doy} {sec_UTC} {F107A} {F107} {AP} {output}'
    args = {
        'min_alt': 0.,
        'max_alt': zmax*1e3,
        'N_samples': N,
        'lat': location[0],
        'lon': location[1],
        'year': date_UTC.year,
        'doy': date_UTC.julday,
        'sec_UTC': sec_UTC,
        'F107A': F107A,
        'F107': F107,
        'AP': AP,
        'output': output,
    }
    cmd = cmd_format.format(**args)
    os.chdir(bin_dir)
    os.system(cmd)
    
    ## Read output file
    msishwm = pd.read_csv(output, skiprows=[0,1], header=[0], delim_whitespace=True)
    msishwm['wx'] = msishwm['w_M[m/s]']
    msishwm = msishwm[['z[m]', 'T[K]', 'wx', 'w_Z[m/s]', 'w_M[m/s]', 'rho[kg/(m^3)]', 'p[Pa]', 'c[m/s]', 'g[m/(s^2)]']]
    msishwm.columns = ['z', 't', 'wx', 'u', 'v', 'rho', 'p', 'c', 'g']
    
    return msishwm
   
import fluids
def get_MSISE(location, date_UTC, zmax, N, use_fluids_lib=False):

    ## Either use fluids library which is already implemented within Python or HWMMSIS Fortran wrapper
    ## fluids library seems to be wrong in its HWM14 implementation
    if use_fluids_lib:
        doy = date_UTC.julday
        alts = np.linspace(0., zmax*1e3, N)
        lats = location[0]
        lons = location[1]
        
        model = pd.DataFrame()
        for z in alts:
            model_base = fluids.atmosphere.ATMOSPHERE_NRLMSISE00(z, latitude=lats, longitude=lons, day=doy)
            model_wind = fluids.atmosphere.hwm14(z, latitude=lats, longitude=lons, day=doy)
            projection_wind = model_wind[1]
            rho = model_base.rho
            c   = fluids.atmosphere.ATMOSPHERE_1976.sonic_velocity(model_base.T)
            P   = model_base.P/100.
            T   = model_base.T
            g   = fluids.atmosphere.ATMOSPHERE_1976.gravity(z)
            
            loc_dict = {
                'z': z, 
                't': T,  
                'wx': projection_wind,
                'u': model_wind[1],
                'v': model_wind[0],
                'rho': rho, 
                'p': P*100., 
                'c': c, 
                'g': g
            }
            
            model = model.append( [loc_dict]  )
            
        """
        model_test = run_MSISHWM_wrapper(location, date_UTC, zmax, N)
        
        model_test = run_MSISHWM_wrapper(location, date_UTC, zmax*1e3, N)
        fig, axs = plt.subplots(1, 2); axs[0].plot(model_test.u, model_test.z, color='tab:red', label='Fortran'); axs[0].plot(model.u, model_test.z, color='tab:blue', label='fluids library'); axs[0].legend(); axs[0].set_title('Zonal at (11.9, -178.99)'); axs[1].plot(model_test.v, model_test.z, color='tab:red'); axs[1].plot(model.v, model_test.z, color='tab:blue'); axs[0].set_xlabel('Wind (m/s)'); axs[0].set_ylabel('Altitude (m)'); axs[1].set_title('Meridional'); axs[1].tick_params(axis='both', which='both', labelleft=False, left=False); plt.show()
        
        bp()
        """
        
    else:
        model = run_MSISHWM_wrapper(location, date_UTC, zmax, N)
   
    return model
   
def get_rotation_matrix(azimuth):

    """
    Get 2d rotation matrix for along a given azimuth
    """

    return np.array(( (np.cos(azimuth), np.sin(azimuth)), (-np.sin(azimuth),  np.cos(azimuth)) ))
   
def compute_effective_velocity_at_z(LON, LAT, sound_veloc, umean, vmean, 
                                    sound_veloc_surface, event_loc):
        
    """
    Compute effective velocity ratio for a given source and velocity model
    """
    
    ## Compute azimuth from source
    """
    Xtemp    = LON - event_loc[1]
    Ytemp    = LAT - event_loc[0]
    costheta = Ytemp/np.sqrt(Xtemp**2+Ytemp**2)
    theta    = np.arccos(costheta)*np.sign(Xtemp)
    """
    from pyproj import Geod
    wgs84_geod = Geod(ellps='WGS84')
    ev_coords = np.repeat(np.array([event_loc]), LON.ravel().size, axis=0)
    az12, az21, distance = wgs84_geod.inv(ev_coords[:,1], ev_coords[:,0], LON.ravel(), LAT.ravel())
    theta = np.radians(az12.reshape(LON.shape))
    #plt.pcolormesh(LON, LAT, distance.reshape(LON.shape)); plt.colorbar(); plt.show()
    
    v_normalized = (sound_veloc + vmean*np.cos(theta) + umean*np.sin(theta))/sound_veloc_surface
    
    """
    quiver_subsample = 8
    umean_ = (0.+umean*1)
    vmean_ = (0.+vmean*1)
    v_normalized_ = (sound_veloc + vmean_*np.cos(theta) + umean_*np.sin(theta))/sound_veloc_surface
    rr_ = np.power(np.add(np.power(umean_[::quiver_subsample,::quiver_subsample],2), np.power(vmean_[::quiver_subsample,::quiver_subsample],2)),0.5)
    fig, axs = plt.subplots(1, 4); axs[0].pcolormesh(LON, LAT, umean_); axs[1].pcolormesh(LON, LAT, sound_veloc/sound_veloc_surface); axs[-1].pcolormesh(LON, LAT, v_normalized_); axs[2].pcolormesh(LON, LAT, np.cos(theta)); axs[-1].quiver(LON[::quiver_subsample,::quiver_subsample], LAT[::quiver_subsample,::quiver_subsample],  umean_[::quiver_subsample,::quiver_subsample]/rr_, vmean_[::quiver_subsample,::quiver_subsample]/rr_, zorder=6, width=0.008); plt.show()
    
    rr = np.power(np.add(np.power(umean[::quiver_subsample,::quiver_subsample],2), np.power(vmean[::quiver_subsample,::quiver_subsample],2)),0.5)
    fig, axs = plt.subplots(1, 4); axs[0].pcolormesh(LON, LAT, umean); axs[1].pcolormesh(LON, LAT, vmean); axs[-1].pcolormesh(LON, LAT, v_normalized); axs[0].scatter(event_loc[1], event_loc[0], marker='x', color='red'); axs[1].scatter(event_loc[1], event_loc[0], marker='x', color='red'); axs[-1].scatter(event_loc[1], event_loc[0], marker='x', color='red'); axs[2].pcolormesh(LON, LAT, np.cos(theta)); axs[-1].quiver(LON[::quiver_subsample,::quiver_subsample], LAT[::quiver_subsample,::quiver_subsample],  umean[::quiver_subsample,::quiver_subsample]/rr, vmean[::quiver_subsample,::quiver_subsample]/rr, zorder=6, width=0.008); plt.show()
    """
    #plt.pcolormesh(LON, LAT, theta); plt.colorbar(); plt.show()
    #lon, lat = np.linspace(-180., 180., 360), np.linspace(-90., 90., 180)
    #LON, LAT = np.meshgrid(lon, lat)
    
    """
    ## Ravel() puts each line after each other
    ## We change the coordinates of each wind vector into the (theta, theta+pi/2) coordinate system to extract the along-azimuth wind strength 
    raveled_winds = np.array([np.ravel(umean), np.ravel(vmean)])
    raveled_theta = theta.ravel()
    rotated_winds = np.array([get_rotation_matrix(theta_).dot(wind) for wind, theta_ in zip(raveled_winds.T, raveled_theta.T)])
    u_unraveled   = rotated_winds[:,0].reshape(*umean.shape)
    v_unraveled   = rotated_winds[:,1].reshape(*vmean.shape)

    ## Compute ratio
    #v_normalized = (sound_veloc + u_unraveled)/sound_veloc_surface
    """
    
    return v_normalized
   
def interpolate_field(xi, yi, lons, lats, field):

    """
    Interpolate a given field in a new xi, yi grid
    """
                
    ## Build grid to display effective sound veloc.
    X_interp, Y_interp = np.meshgrid(xi, yi)
    X, Y = np.meshgrid(lons, lats)
    
    X1d = X.ravel()
    Y1d = Y.ravel()
    coords = np.column_stack((X1d,Y1d))
    
    ## If only one latitude, we duplicate arrays for interpolation                                
    if field.shape[0] < 2:
        field = np.concatenate((field, field, field))
        
        lat  = lats[0]
        lats_ = np.array([lat-1., lat, lat+1.])
        
        X, Y = np.meshgrid(lons, lats_)
        X1d = X.ravel()
        Y1d = Y.ravel()
        coords = np.column_stack((X1d,Y1d))
    
    z = field.ravel()
    
    field_interpolated = griddata(coords, z, (X_interp, Y_interp), method='cubic') 
    
    return X_interp, Y_interp, field_interpolated   

def get_levels_to_height():

    return pd.read_csv('/staff/quentin/Documents/Projects/generalroutines/pressure_height_relation_137_levels.csv')

def get_height_from_pressure_level(levels):

    levels_to_height = get_levels_to_height()
    heights          = [levels_to_height['Height(m)'].iloc[ abs(levels_to_height['m']-i).idxmin() ] for i in levels]
    return heights

def get_sound_veloc_from_temperature(temperature):

    """
    Use standard atmosphere model to build sound velocity from temperature
    """

    sound_veloc = 0.5144444*643.855*(temperature/273.15)**0.5
    return sound_veloc

def get_properties_nc(wnds):
        
    ## Find data in files
    lons   = wnds.variables['longitude'][:] 
    if lons.max() > 180:
            lons -= 360.
    lats   = wnds.variables['latitude'][:]
    levels = wnds.variables['level'][:]
    times  = wnds.variables['time'][:]
    
    heights = get_height_from_pressure_level(levels)
    
    return lats, lons, heights, levels, times

def read_one_netcdf(file):

    wnds = Dataset(file, mode='r')
    lats, lons, heights, levels, times = get_properties_nc(wnds)
    zonal      = wnds.variables['u'][:].filled()
    meridonial = wnds.variables['v'][:].filled()
    temperature = wnds.variables['t'][:].filled()
    
    return lats, lons, heights, levels, times, zonal, meridonial, temperature

def get_surface_field(heights, field):

    """
    Extract field at the surface
    """

    ## Coordinate of the surface
    iz0 = np.argmin( np.abs( np.array(heights) - 0. ) ) 

    N_components_field = len(field.shape)
    if N_components_field == 5:
        field_surface = np.mean(field[:,:,iz0,:,:], axis=(0,1))
    else:
        field_surface = np.mean(field[:,iz0,:,:], axis=(0))

    return field_surface

def get_avg_model_time_and_altitude(uwnd_amp, vwnd_amp, sound_veloc, i0, iend):

    """
    Average models over time
    """

    N_components_field = len(uwnd_amp.shape)
    if N_components_field == 5:
        uwnd_amp_avg = np.mean(uwnd_amp[:,:,i0:iend,:,:], axis=(0,1,2))
        vwnd_amp_avg = np.mean(vwnd_amp[:,:,i0:iend,:,:], axis=(0,1,2))
        sound_veloc_avg = np.mean(sound_veloc[:,:,i0:iend,:,:], axis=(0,1,2))
    else:
        uwnd_amp_avg = np.mean(uwnd_amp[:,i0:iend,:,:], axis=(0,1))
        vwnd_amp_avg = np.mean(vwnd_amp[:,i0:iend,:,:], axis=(0,1))
        sound_veloc_avg = np.mean(sound_veloc[:,i0:iend,:,:], axis=(0,1))
        
    return uwnd_amp_avg, vwnd_amp_avg, sound_veloc_avg

def project_wind_along_azimuth(meridional, zonal, baz_ref_station):
    
    """
    Project meridonial and zonal winds along a given azimuth
    """
    
    baz_ref_station_rad = np.radians(baz_ref_station)
    projection_wind = zonal*np.sin(baz_ref_station_rad) + meridional*np.cos(baz_ref_station_rad)
    return projection_wind

class atmos_model():
    
    def __init__(self, source, receiver, time, max_height, nc_file):
        
        ## Save input parameters
        self.time       = time
        self.max_height = max_height
        self.nc_file    = nc_file
        self.source, self.receiver = source, receiver
        
        ## Extract lat/lon and atmospheric data from netcdf
        self.lats, self.lons, self.heights, self.levels, self.times, \
            self.zonal, self.meridonial, self.temperature = read_one_netcdf(self.nc_file)
        
    def _update_atmos_model(self, N=1000, projection=False, ref_projection={}, smooth_model=True, smooth_radius=7.5, add_MSISE_var=True):
    
        #print('-->', self.list_of_points)
    
        self.updated_model = pd.DataFrame()
        for ilocation in range(self.list_of_points.shape[0]):
            
            location = (self.list_of_points[ilocation, 1], self.list_of_points[ilocation, 0])
            distance = self.list_of_points[ilocation, 2]
            
            if np.max(self.heights) >= self.max_height*1e3 and not add_MSISE_var:
            
                MSISE = pd.DataFrame()
                new_heights = np.linspace(0., self.max_height*1e3, N)
                for variable in self.external_model_interpolated:
                    field = self.external_model_interpolated[variable][:, ilocation]
                    f = interpolate.interp1d(self.heights, field, kind='cubic', fill_value='extrapolate')
                    MSISE[variable] = f(new_heights)
                MSISE['c'] = np.sqrt(401.87430086589046*MSISE['t'])
                MSISE['z'] = new_heights
                    
            else:
            
                MSISE = get_MSISE(location, self.time, self.max_height, N)
                for variable in self.external_model_interpolated:
                    field = self.external_model_interpolated[variable][:, ilocation]
                    f = interpolate.interp1d(self.heights, field, kind='cubic')
                    new_heights = MSISE.loc[(MSISE['z'] <= self.heights[0]) & (MSISE['z'] >= self.heights[-1]), 'z'].values
                    field_interpolated = f(new_heights)
                    MSISE.loc[(MSISE['z'] <= self.heights[0]) & (MSISE['z'] >= self.heights[-1]), variable] = field_interpolated
                
                    if smooth_model:
                        MSISE_out_smooth = MSISE.loc[(MSISE.z <= np.max(self.heights)-smooth_radius*1e3) | (MSISE.z >= np.max(self.heights)+smooth_radius*1e3), :]
                        f = interpolate.interp1d(MSISE_out_smooth.z.values, MSISE_out_smooth[variable].values, kind='cubic')
                        #MSISE_test = MSISE.copy()
                        MSISE[variable] = f(MSISE.z.values)
                       
                if projection:
                    #baz_ref_station = gps2dist_azimuth(location[0], location[1], ref_projection['lat'], ref_projection['lon'])[2]
                    #baz_ref_station = np.radians(baz_ref_station)
                    #projection_wind = MSISE['v'].values*np.cos(baz_ref_station) + MSISE['u'].values*np.sin(baz_ref_station)
                    projection_wind = project_wind_along_azimuth(MSISE['v'].values, MSISE['u'].values, self.baz_ref_station)
                    MSISE['wx'] = projection_wind
        
            MSISE = MSISE.assign(lat = location[0], lon = location[1], distance = distance)
            self.updated_model = self.updated_model.append( MSISE.copy() )
                
        self.updated_model.reset_index(drop=True, inplace=True)
        
    def build_effective_velocity_map(self, altitude_layers, nb_lats=100, nb_lons=100):
        
        self.altitude_layers = altitude_layers
        self.sound_veloc = get_sound_veloc_from_temperature(self.temperature)
        
        df_v_normalized = pd.DataFrame()
        ## Loop over each layer
        self.ceffs = []
        for altbounds in self.altitude_layers:
            
            LON, LAT, umean, vmean, sound_veloc, sound_veloc_surface = \
                self._build_average_maps(altbounds, nb_lats=nb_lats, nb_lons=nb_lons)
            effective_velocity = \
                compute_effective_velocity_at_z(LON, LAT, sound_veloc, umean, vmean, 
                                                sound_veloc_surface, self.source)
            
            self.ceffs.append( {'altbounds': altbounds, 'LON': LON, 'LAT': LAT, 'ceff': effective_velocity} )
        
    def construct_effective_velocity_map(self, altitude_layers, dir_figures, name_event, stations, 
                                        dlat=0.5, dlon=1., levels=[1.], vmin_ceff=-1, vmax_ceff=-1,
                                        global_map_degrees_offset = 10.,
                                        image_extension='pdf'):
        
        """
        Build and plot an effective velocity map for a list of altitude layers
        """
        
        self.dir_figures = dir_figures
        self.name_event  = name_event
        self.build_effective_velocity_map(altitude_layers, nb_lats=100, nb_lons=100)
        
        ## Loop over each layer
        for altbounds, ceff in zip(self.altitude_layers, self.ceffs):
            
            LON, LAT, effective_velocity = ceff['LON'], ceff['LAT'], ceff['ceff']
            self.ceffs.append( {'altbounds': altbounds, 'LON': LON, 'LAT': LAT, 'ceff': effective_velocity} )
        
            """
            profiles = {
        'z': self.updated_model.z.values/1e3,
        'u': self.updated_model.v.values,
        'v': self.updated_model.u.values,
        't': self.updated_model.c.values,
        'uv-projected': construct_atmospheric_model.project_wind_along_azimuth(meridonial, zonal, baz_ref_station),
            """
            plotting_routines.plot_map_effective_velocity(effective_velocity, LON[0,:], LAT[:,0], 
                                        altbounds, self.source, self.receiver, 
                                        umean, vmean, self.updated_model.z.values/1e3, 
                                        self.updated_model.v.values, self.updated_model.u.values, 
                                        self.updated_model.c.values, self.baz_ref_station, self.time, 
                                        self.dir_figures, self.name_event, station_data=stations, fsz=10, 
                                        dlat=dlat, dlon=dlon, 
                                        levels=levels, vmin_ceff=vmin_ceff, vmax_ceff=vmax_ceff,
                                        global_map_degrees_offset = global_map_degrees_offset,
                                        image_extension=image_extension)
        
        
    def _build_average_maps(self, altbounds, nb_lats = 100, nb_lons = 100):
    
        ## Compute vertically averaged fields 
        alt0   = np.argmin( np.abs( np.array(self.heights) - altbounds[1] ) ) 
        altend = np.argmin( np.abs( np.array(self.heights) - altbounds[0] ) )
        uwnd_amp_avg, vwnd_amp_avg, sound_veloc_avg = \
            get_avg_model_time_and_altitude(self.meridonial, self.zonal, 
                                            self.sound_veloc, alt0, altend)
        
        ## Extract surface sound velocity
        sound_veloc_surface = get_surface_field(self.heights, self.sound_veloc)
        
        ## Interpolate fields
        xi = np.linspace(self.lons[0], self.lons[-1], nb_lons)
        yi = np.linspace(self.lats[0], self.lats[-1], nb_lats)
        LON, LAT, vmean_interpolated = interpolate_field(xi, yi, self.lons, self.lats, vwnd_amp_avg)
        _, _, umean_interpolated = interpolate_field(xi, yi, self.lons, self.lats, uwnd_amp_avg)
        _, _, sound_veloc_interpolated = interpolate_field(xi, yi, self.lons, self.lats, sound_veloc_avg)
        _, _, sound_veloc_surface_interpolated = interpolate_field(xi, yi, self.lons, self.lats, sound_veloc_surface)
        
        return LON, LAT, umean_interpolated, vmean_interpolated, sound_veloc_interpolated, sound_veloc_surface_interpolated
        
    def construct_profiles(self, number_points, number_altitudes, 
                           type='specfem', range_dependent=False, 
                           projection=False, ref_projection={}, handle_ensembles=-1,
                           list_of_points=np.array([]), offset = 7.):
    
        self.number_points = number_points
        
        ## Compute entire region
        print('List of points')
        self.list_of_points = list_of_points
        if list_of_points.size == 0:
            self.baz_ref_station = gps2dist_azimuth(self.source[0], self.source[1], 
                                                    self.receiver[0], self.receiver[1])[1]#ref_projection['lat'], ref_projection['lon'])[2]
            
            ## Find discretization points between source and receiver
            self.list_of_points = build_locations(self.source, self.receiver, self.number_points)
        else:
            self.number_points = list_of_points.size
            self.list_of_points = list_of_points
        
        ## Interpolate profiles along the source-receiver slice
        print('Before interpolation')
        self.external_model_interpolated = \
            interpolate_external_model(self.list_of_points, self.lats, self.lons, 
                                       self.heights, self.levels, self.zonal, 
                                       self.meridonial, self.temperature, kind_1d_interp='cubic', 
                                       default_offset_interp=1e-5, offset = offset, handle_ensembles=handle_ensembles)
        
        # Check if crashed interpolation
        #if self.external_model_interpolated['u'][10,0] == 0.:
        #    bp()
        print('Before update')
        
        self._update_atmos_model(N=number_altitudes, projection=projection, ref_projection=ref_projection)
        #plot_profiles(dir, self.updated_model)
        
        if not range_dependent:
            self.updated_model = self.updated_model.groupby('z').mean().reset_index()
            
##########################
if __name__ == '__main__':
    time             = UTCDateTime(2020, 5, 18, 1, 11, 57)
    max_height       = 120. # km
    number_points    = 5
    number_altitudes = 1000
    source, receiver = (67.5, 21.8), (69., 23.2)
    
    #nc_file = '/staff/quentin/Documents/Codes/specfem-dg/EXAMPLES/model_ERA5_2020-05-18_01.11.57_67.62943567881625_68.03586432118374_20.078335678816252_20.599964321183748.nc'
    nc_file = '/staff/quentin/Documents/Projects/test_output/model_ECMWF_2021-03-31_00.00.00_73.0_18.0_67.0_26.0.nc'
    one_atmos_model = atmos_model(source, receiver, time, max_height, nc_file)
    one_atmos_model.construct_profiles(number_points, number_altitudes, type='specfem', range_dependent=False)
    
    dir_figures = '../'
    name_event  = 'test'
    altitude_layers = [[0., 10000.], [10000., 20000.]]
    stations = [
        {'station': 'test', 'Lat': 69., 'Lon': 23.2 },
    ]
    one_atmos_model.construct_effective_velocity_map(altitude_layers, dir_figures, name_event, stations)
    bp()