import tqdm
import pandas as pd
import numpy as np
import os
import obspy # obspy 1.3.1
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import ifft, fft, fftfreq
import seaborn as sns # seaborn 0.11.1
import scipy
from pyrocko import moment_tensor as mtm
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import patheffects
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

STRENGTH_SMOOTHING = 1.0

path_effects = [patheffects.withStroke(linewidth=3, foreground="w")]

def get_coefs_fundamental_mt(mt_target_in):

    ## Convert coordinates USE -> NED
    mt_target = mt_target_in.copy()
    if 'Mrr' in mt_target.keys():
        mt_target = dict(
            Mnn=mt_target_in['Mtt'],
            Mne=-mt_target_in['Mtp'],
            Mnd=mt_target_in['Mrt'],
            Mee=mt_target_in['Mpp'],
            Med=-mt_target_in['Mrp'],
            Mdd=mt_target_in['Mrr'],
        )

    ## e.g., mt_target = {'Mnn': 0.5, 'Mne': 0.3, 'Mnd': 0.2, 'Mee': -0., 'Med': 1., 'Mdd': 1.,  }
    a = np.array([[0,0,0,-1,0,1], [1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,0,0,-1,1], [0,0,-1,0,0,0], [0,0,0,1,1,1]])
    b = np.array(list(mt_target.values()))
    results = np.linalg.solve(a, b)
    coefs = {}
    for icoef, coef in enumerate(results):
        coefs[f'M{icoef+1}'] = coef

    inverted = np.array([a[:,imt]*results[imt] for imt in range(a.shape[1])]).sum(axis=0)

    print('----------------')
    print(f'Original mt: {np.array(list(mt_target.values()))}')
    print(f'Inverted mt: {inverted}')
    print(f'Coefficients: {coefs}')

    return coefs

def generate_noise_timeseries(pd_F, tmax, freqmin, freqmax, scale=1., dt_new=-1.):
    
    Fourier_spectrum = pd_F.spectrum.values
    dt = pd_F.dt.iloc[0]

    # Compute amplitude spectrum
    amplitude_spectrum = (Fourier_spectrum)**1
    
    # Generate random phase spectrum for positive frequencies
    positive_phase_spectrum = np.random.uniform(0, 2*np.pi, Fourier_spectrum.shape[0]//2 + 1)

    # Create conjugate symmetric phase spectrum
    phase_spectrum = np.concatenate([positive_phase_spectrum, -positive_phase_spectrum[-2::-1]])

    # Combine amplitude and phase to form complex spectrum
    complex_spectrum = amplitude_spectrum * np.exp(1j * phase_spectrum)

    # Apply inverse Fourier transform
    time_signal = np.fft.ifft(complex_spectrum).real
    
    tr = obspy.Trace()
    tr.data = time_signal
    tr.stats.delta = dt

    if dt_new > 0:
        tr.resample(1./dt_new)

    tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
    tr.trim(endtime=tr.stats.starttime+tmax)
    tr.data *= scale

    return tr

def plot_waveforms(st, all_arrivals=pd.DataFrame(), figsize=(4,8), normalize_individual_waveform=False, normalize_spectro_per_freq=False, sort_by_distance=False, freq_min=1e-2, freq_max=10., id_Sxx=-1, q_display=0.995, vel_IS=0.34, vel_S=3.5, use_rel_time=-1, time_min=0., time_max=-1., show_only_km_label=False, **kwargs):
    
    multiple_waveforms = False
    h_total = 4
    h_spectro = 1
    if len(st) > 1:
        multiple_waveforms = True
        h_total = 8
        h_spectro = 2
    
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(h_total, 1)
    
    ax = fig.add_subplot(grid[h_spectro:,:])
    global_norm = 1.
    if not normalize_individual_waveform:
        global_norm = np.max([tr.data.max() for tr in st])
    
    l_distances = np.arange(len(st))
    if sort_by_distance:
        l_distances = [tr.stats.distance for tr in st]
    index_sort = np.argsort(l_distances)
        
    #for itr, tr in enumerate(st):
    min_offset = -1

    xlim = st[0].times().max()
    if time_max > -1:
        xlim = time_max

    l_colors = {}
    colors = sns.color_palette('tab10', n_colors=all_arrivals.phase.unique().size)
    for icolor, phase in enumerate(all_arrivals.phase.unique()):
        l_colors[phase] = colors[icolor]
    
    labeled_phases = []
    for offset, itr in enumerate(index_sort):
        tr = st[itr]
        if normalize_individual_waveform:
            global_norm = tr.data.max()
        color = sns.color_palette('rocket', n_colors=len(st))[offset]

        
        offset_rel_time = 0.
        if use_rel_time > 0:
            offset_rel_time = tr.stats.distance/use_rel_time
        min_offset = max(min_offset, offset_rel_time)
        
        ax.plot(tr.times()-offset_rel_time, tr.data/global_norm+offset, color=color)

        if all_arrivals.shape[0] > 0:

            label_P, label_S, label_RW = {}, {}, {}
            
            offset_time = 0.
            if tr.stats.altitude > 0:
                offset_time += tr.stats.altitude/vel_IS 
                print(tr.stats.station, tr.stats.distance, tr.stats.altitude, vel_IS, offset_time)

            phases = all_arrivals.loc[abs(all_arrivals.distance-tr.stats.distance)==abs(all_arrivals.distance-tr.stats.distance).min()]
            phases.sort_values(by='time', inplace=True)
            #phases_P = phases.loc[(phases.phase=='P')|((phases.phase=='Pn'))]
            
            phase_times = []
            min_diff_time_phases = 1.
            for _, phase in phases.iterrows():
                if phase.time+offset_time >= tr.times().max():
                    continue
                label_phase = {}

                if phase_times:
                    if abs(np.array(phase_times)-phase.time).min()<min_diff_time_phases:
                        continue
                phase_times.append(phase.time)

                if (not phase.phase in labeled_phases):
                    label_phase['label'] = phase.phase
                    labeled_phases.append(phase.phase)
                    
                ax.scatter(phase.time+offset_time, offset, color=l_colors[phase.phase], marker='|', **label_phase)
            """
            phases_P = phases.loc[phases.phase.str.contains('P')|phases.phase.str.contains('p')]
            phases_P = phases_P.loc[phases_P.time==phases_P.time.min()].iloc[0]
            if itr == 0:
                label_P['label'] = phases_P.phase
            ax.scatter(phases_P.time+offset_time, offset, color='red', marker='|', **label_P)
            #phases_S = phases.loc[(phases.phase=='S')|((phases.phase=='Sn'))]
            phases_S = phases.loc[phases.phase.str.contains('S')|phases.phase.str.contains('s')]
            phases_S = phases_S.loc[phases_S.time==phases_S.time.min()].iloc[0]
            if itr == 0:
                label_S['label'] = phases_S.phase
            ax.scatter(phases_S.time+offset_time, offset, color='green', marker='|', **label_S)
            """
            
            phases_RW = tr.stats.distance/(vel_S*0.92)# vel 3.5
            if itr == 0:
                label_RW['label'] = 'RW'
            ax.scatter(phases_RW+offset_time, offset, color='blue', marker='|', **label_RW)

        text = f'{tr.stats.distance:.1f} km'
        if not show_only_km_label:
            text = f'{tr.stats.station} - {text}'
        
        ax.text(xlim, offset, text, ha='left', va='bottom', path_effects=path_effects, transform=ax.transData)
    
    ax.set_xlim([time_min-min_offset, xlim]) 
    ax.set_ylabel('Amplitude')

    unit = 'm/s'
    if st[0].stats.altitude > 0:
        unit = 'Pa'

    ax.text(1., 0., f'Maximum: {global_norm:.2e} {unit}', ha='right', va='bottom', path_effects=path_effects, transform=ax.transAxes)
    if all_arrivals.shape[0] > 0:
        ax.legend(frameon=True, loc='upper left')
    
    xlabel = f'Time (s) since {tr.stats.starttime}'
    if use_rel_time > 0:
        xlabel = f'Rel. time (s, t-d/v, v={use_rel_time:.2f} km/s) since {tr.stats.starttime}'
        
    ax.set_xlabel(xlabel)
    ax.tick_params(axis='both', which='both', left=False, labelleft=False,)
    
    ax_spectro = fig.add_subplot(grid[:h_spectro,:], sharex=ax)
    tr = st[index_sort[id_Sxx]]
    f, t, Sxx = signal.spectrogram(tr.data, fs=1./tr.stats.delta, **kwargs); 
    #ax_spectro.plot(tr.times(), tr.data/global_norm+itr)

    if normalize_spectro_per_freq:
        Sxx /= Sxx.max(axis=1, keepdims=True)

    opt_vmax = {}
    if q_display > 0:
        opt_vmax['vmax'] = np.quantile(Sxx, q=q_display)
        
    offset_rel_time = 0.
    if use_rel_time > 0:
        offset_rel_time = tr.stats.distance/use_rel_time
        
    ax_spectro.pcolormesh(t-offset_rel_time, f, Sxx, shading='auto', cmap='cividis', **opt_vmax);
    #ax_spectro.set_yscale('log')
    ax_spectro.set_ylim([freq_min, freq_max])
    ax_spectro.set_ylabel('Freq. (Hz)')
    ax_spectro.tick_params(axis='both', which='both', labelbottom=False,)
    
    axs = [ax, ax_spectro]
    fig.align_ylabels(axs)
    fig.subplots_adjust(hspace=0., right=0.85)

def dispcurve_penaltyfunc(vgarray, amplarray, strength_smoothing=STRENGTH_SMOOTHING):
    """
    Objective function that the vg dispersion curve must minimize.
    The function is composed of two terms:

    - the first term, - sum(amplitude), seeks to maximize the amplitudes
      traversed by the curve
    - the second term, sum(dvg**2) (with dvg the difference between
      consecutive velocities), is a smoothing term penalizing
      discontinuities

    *vgarray* is the velocity curve function of period, *amplarray*
    gives the amplitudes traversed by the curve and *strength_smoothing*
    is the strength of the smoothing term.

    @type vgarray: L{numpy.ndarray}
    @type amplarray: L{numpy.ndarray}
    """
    # removing nans
    notnan = ~(np.isnan(vgarray) | np.isnan(amplarray))
    vgarray = vgarray[notnan]

    # jumps
    dvg = vgarray[1:] - vgarray[:-1]
    sumdvg2 = np.sum(dvg**2)

    # amplitude
    sumamplitude = amplarray.sum()

    # vg curve must maximize amplitude and minimize jumps
    return -sumamplitude + strength_smoothing*sumdvg2

def optimize_dispcurve(amplmatrix, velocities, vg0, periodmask=None,
                       strength_smoothing=STRENGTH_SMOOTHING):
    """
    Optimizing vel curve, i.e., looking for curve that really
    minimizes *dispcurve_penaltyfunc* -- and does not necessarily
    ride any more through local maxima

    Returns optimized vel curve and the corresponding
    value of the objective function to minimize

    @type amplmatrix: L{numpy.ndarray}
    @type velocities: L{numpy.ndarray}
    @rtype: L{numpy.ndarray}, float
    """
    if np.any(np.isnan(vg0)):
        raise Exception("Init velocity array cannot contain NaN")

    nperiods = amplmatrix.shape[0]

    # function that returns the amplitude curve
    # a given input vel curve goes through
    ixperiods = np.arange(nperiods)
    amplcurvefunc2d = RectBivariateSpline(ixperiods, velocities, amplmatrix, kx=1, ky=1)
    amplcurvefunc = lambda vgcurve: amplcurvefunc2d.ev(ixperiods, vgcurve)

    def funcmin(varray):
        """Objective function to minimize"""
        # amplitude curve corresponding to vel curve
        if not periodmask is None:
            return dispcurve_penaltyfunc(varray[periodmask],
                                         amplcurvefunc(varray)[periodmask],
                                         strength_smoothing=strength_smoothing)
        else:
            return dispcurve_penaltyfunc(varray,
                                         amplcurvefunc(varray),
                                         strength_smoothing=strength_smoothing)
            
    bounds = nperiods * [(min(velocities) + 0.1, max(velocities) - 0.1)]
    method = 'SLSQP'  # methods with bounds: L-BFGS-B, TNC, SLSQP
    resmin = minimize(fun=funcmin, x0=vg0, method=method, bounds=bounds)
    vgcurve = resmin['x']
    # _ = funcmin(vgcurve, verbose=True)

    return vgcurve, resmin['fun']

def FTAN(x, dt, periods, alpha, phase_corr=None):
    """
    Frequency-time analysis of a time series.
    Calculates the Fourier transform of the signal (xarray),
    calculates the analytic signal in frequency domain,
    applies Gaussian bandpass filters centered around given
    center periods, and calculates the filtered analytic
    signal back in time domain.
    Returns the amplitude/phase matrices A(f0,t) and phi(f0,t),
    that is, the amplitude/phase function of time t of the
    analytic signal filtered around period T0 = 1 / f0.

    See. e.g., Levshin & Ritzwoller, "Automated detection,
    extraction, and measurement of regional surface waves",
    Pure Appl. Geoph. (2001) and Bensen et al., "Processing
    seismic ambient noise data to obtain reliable broad-band
    surface wave dispersion measurements", Geophys. J. Int. (2007).

    @param dt: sample spacing
    @type dt: float
    @param x: data array
    @type x: L{numpy.ndarray}
    @param periods: center periods around of Gaussian bandpass filters
    @type periods: L{numpy.ndarray} or list
    @param alpha: smoothing parameter of Gaussian filter
    @type alpha: float
    @param phase_corr: phase correction, function of freq
    @type phase_corr: L{scipy.interpolate.interpolate.interp1d}
    @rtype: (L{numpy.ndarray}, L{numpy.ndarray})
    """

    # Initializing amplitude/phase matrix: each column =
    # amplitude function of time for a given Gaussian filter
    # centered around a period
    amplitude = np.zeros(shape=(len(periods), len(x)))
    phase = np.zeros(shape=(len(periods), len(x)))

    # Fourier transform
    Xa = fft(x)
    # aray of frequencies
    freq = fftfreq(len(Xa), d=dt)

    # analytic signal in frequency domain:
    #         | 2X(f)  for f > 0
    # Xa(f) = | X(f)   for f = 0
    #         | 0      for f < 0
    # with X = fft(x)
    Xa[freq < 0] = 0.0
    Xa[freq > 0] *= 2.0

    # applying phase correction: replacing phase with given
    # phase function of freq
    if phase_corr:
        # doamin of definition of phase_corr(f)
        minfreq = phase_corr.x.min()
        maxfreq = phase_corr.x.max()
        mask = (freq >= minfreq) & (freq <= maxfreq)

        # replacing phase with user-provided phase correction:
        # updating Xa(f) as |Xa(f)|.exp(-i.phase_corr(f))
        phi = phase_corr(freq[mask])
        Xa[mask] = np.abs(Xa[mask]) * np.exp(-1j * phi)

        # tapering
        taper = cosTaper(npts=mask.sum(), p=0.05)
        Xa[mask] *= taper
        Xa[~mask] = 0.0

    # applying narrow bandpass Gaussian filters
    for iperiod, T0 in enumerate(periods):
        # bandpassed analytic signal
        f0 = 1.0 / T0
        Xa_f0 = Xa * np.exp(-alpha * ((freq - f0) / f0) ** 2)
        # back to time domain
        xa_f0 = ifft(Xa_f0)
        # filling amplitude and phase of column
        amplitude[iperiod, :] = np.abs(xa_f0)
        phase[iperiod, :] = np.angle(xa_f0)

    return amplitude, phase

def compute_one_cg(crustal_thickness, periods, max_mode, vs_crust=3.5, vs_mantle=4.4):
    
    from disba import GroupDispersion
    
    # Velocity model
    # thickness, Vp, Vs, density
    # km, km/s, km/s, g/cm3
    velocity_model = np.array([
       #[5., 5.00, 1.50, 2.80],
       [crustal_thickness, 6.00, vs_crust, 2.80],
       [1000.0, 7.50, vs_mantle, 3.30],
    ])

    # Compute the 3 first Rayleigh- and Love- wave modal dispersion curves
    # Fundamental mode corresponds to mode 0
    pd_g = GroupDispersion(*velocity_model.T)
    cgrs = [pd_g(periods, mode=i, wave="rayleigh") for i in range(max_mode)]
    return cgrs

def extract_dispersion_curve(waveform_in, periods, vels, alpha=20., max_percentage=0.3, vg0=None, trim_after=-1., window_around_arrival={'time': 2., 'duration': 2., 'taper': 0.1}, norm_ftan=False, vel_acoustic=0.335):

    waveform = waveform_in.copy()
    if trim_after > 0:
        waveform.trim(endtime=waveform.stats.starttime+trim_after)
    waveform.taper(max_percentage=max_percentage)
    t = waveform.times()
    data = waveform.data
    inds = t>=0.
    t = t[inds]-t[inds].min()
    data = data[inds]

    if window_around_arrival:
        iarrival = np.argmin(abs(t-window_around_arrival['time']))
        iduration = np.argmin(abs(t[t>=0]-window_around_arrival['duration']))
        inds = np.arange(max(iarrival-iduration//2, 0), min(data.size, iarrival+iduration//2))
        w = signal.windows.tukey(data[inds].size, alpha=window_around_arrival['taper'])
        data_c = np.zeros_like(data)
        data_c[inds] = data[inds]*w
        data = data_c 
        
    toffset_atmos = 0.
    if waveform.stats.altitude > 0:
        toffset_atmos = waveform.stats.altitude/vel_acoustic
        t -= toffset_atmos
        data = data[t>=0]
        t = t[t>=0]
        
    vels_orig = (waveform.stats.distance / t[t!=0])[::-1]  # velocities
    dt = waveform.stats.delta

    amplitude, _ = FTAN(data, dt, periods, alpha)
    zampl = amplitude[:, t!=0][:, ::-1]

    ampl_interp_func = RectBivariateSpline(periods, vels_orig, zampl)
    ampl_resampled = ampl_interp_func(periods, vels)
    
    if not vg0:
        imax, jmax = np.where(ampl_resampled==ampl_resampled.max())
        vg0 = vels[jmax[0]]*np.ones_like(periods)
        print(f'Use vg_0={vels[jmax[0]]:.2f} km/s')
    elif vg0 == 'disba':
        crustal_thickness = 11.
        cgrs = compute_one_cg(crustal_thickness, periods, 1, vs_crust=3.5, vs_mantle=4.4)
        vg0 = cgrs[0].velocity
        
    ampl_resampled_orig = ampl_resampled.copy()
    if norm_ftan:
        ampl_resampled /= ampl_resampled.max(axis=1, keepdims=True)
        
    vgcurve, misfit = optimize_dispcurve(ampl_resampled, vels, vg0)
        
    return t, toffset_atmos, data, vgcurve, misfit, ampl_resampled, ampl_resampled_orig

def plot_ftan(waveform, time_data, toffset_atmos,  data, periods, vels, vgcurve, ampl_resampled, ampl_resampled_orig, figsize=(5,3), fontsize=12., norm_ftan=False, show_theoretical=True, crustal_thickness=25., max_mode=2):
    
    if show_theoretical:
        
        """
        from disba import GroupDispersion, GroupSensitivity, EigenFunction

        # Velocity model
        # thickness, Vp, Vs, density
        # km, km/s, km/s, g/cm3
        velocity_model = np.array([
           #[5., 5.00, 1.50, 2.80],
           [crustal_thickness, 6.00, 3.50, 2.80],
           [1000.0, 7.50, 4.4, 3.30],
        ])
        
        
        # Compute the 3 first Rayleigh- and Love- wave modal dispersion curves
        # Fundamental mode corresponds to mode 0
        pd_g = GroupDispersion(*velocity_model.T)
        cgrs = [pd_g(periods, mode=i, wave="rayleigh") for i in range(max_mode)]
        """
        """
        ps = GroupSensitivity(*velocity_model.T)
        parameters = ["thickness", "velocity_s"]
        periods_kernel = [periods.min(), periods.max()]
        kernels = {}
        for period in periods_kernel:
            kernels[period] = [ps(period, mode=0, wave="rayleigh", parameter=parameter) for parameter in parameters]
            print(kernels[period])
        """
        #eigf = EigenFunction(*velocity_model.T)
        #eigr = eigf(0.1, mode=0, wave="rayleigh")
        
        cgrs = compute_one_cg(crustal_thickness, periods, max_mode, vs_crust=3.5, vs_mantle=4.4)
    
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(4, 1)
    axs = []
    ax = fig.add_subplot(grid[:3,:])
    axs.append(ax)
    norm_amp = ampl_resampled
    if norm_ftan:
        norm_amp /= norm_amp.max(axis=1, keepdims=True)
    ax.pcolormesh(periods, vels, norm_amp.T, shading='auto')
    X, Y = np.meshgrid(periods, vels)
    CS1 = ax.contour(X, Y, ampl_resampled_orig.T, levels=[np.quantile(ampl_resampled_orig, q=0.75)], )
    fmt = {}
    strs = ['$75\%$']
    for l, s in zip(CS1.levels, strs):
        fmt[l] = s
    ax.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=10)
    
    ax.plot(periods, vgcurve, color='red', label='$c_g$')
    ax.text(periods[-1], vels.max(), f'{waveform.stats.distance:.2f} km', path_effects=path_effects, ha='right', va='top', transform=ax.transData)
    ax.set_xlabel('Period (s)', fontsize=fontsize)
    ax.set_ylabel('Velocity (km/s)', fontsize=fontsize)
    ax.tick_params(axis='both', which='both', labelbottom=False, bottom=False, labeltop=True, top=True)
    ax.xaxis.set_label_position('top')
    
    if show_theoretical:
        colors = sns.color_palette('rocket', n_colors=len(cgrs))
        imode = 0
        for color, cgr in zip(colors, cgrs):
            ax.plot(cgr.period, cgr.velocity, color=color, linestyle=':', label=f'theo $c_g$ mode {imode}')
            imode += 1
    ax.legend(loc='lower right')
        
    ax = fig.add_subplot(grid[3,:])
    axs.append(ax)
    nt0 = waveform.times()>0
    sc = ax.scatter(waveform.times()[nt0], waveform.data[nt0], c=waveform.stats.distance/(waveform.times()[nt0]-toffset_atmos), vmin=vels.min(), vmax=vels.max(), s=10, cmap='magma')
    ax.plot(time_data+toffset_atmos, data, color='red', linestyle=':', label='Tapered signal')
    axins = inset_axes(ax, width="3%", height="80%", loc='lower left', bbox_to_anchor=(1.03, 0.1, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = plt.colorbar(sc, cax=axins, orientation='vertical', extend='both')
    cbar.ax.set_ylabel('Velocity (km/s)')
    ax.set_xlabel('Time (s)', fontsize=fontsize)
    ax.set_ylabel('Pressure (Pa)', fontsize=fontsize)
    ax.legend()
    ax.set_xlim([time_data.min()+toffset_atmos, time_data.max()+toffset_atmos])
    
    fig.align_ylabels(axs)
    fig.subplots_adjust(right=0.8)

def extrapolate_TL(waveform_data, dists_new=np.linspace(0., 1000, 100), plot=True, norm_factor=1e8):
    
    dists = np.array([waveform.stats.distance for waveform in waveform_data])
    isort = np.argsort(dists)
    dists = dists[isort]
    max_amp = np.array([abs(waveform.data).max() for waveform in waveform_data])[isort]
    #max_amp /= m0
    max_amp /= norm_factor
    coefs = np.polyfit(dists, np.log(max_amp), 1, )
    sigma = np.ones_like(dists)
    func = lambda dist,a,b,c,d: a*(dists.min()/dist)**b + c*(dists.min()/dist)**d
    coefs, _ = scipy.optimize.curve_fit(func,  dists,  max_amp, sigma=sigma, p0=(np.exp(coefs[1]), coefs[0], 0.1, 2.))
    a,b,c,d = coefs
    fitted = lambda dist: func(dist,a,b,c,d)

    if plot:
        plt.figure()
        plt.plot(dists, max_amp, label='data')
        plt.xlabel('Distance (km)')
        plt.ylabel('Pressure (Pa)')

        dists_new = np.linspace(0., 1000, 100)
        plt.plot(dists_new, fitted(dists_new), label=f'${{{a:.2e}}}(d_0/d)^{{{b:.2f}}} + {{{c:.2e}}}(d_0/d)^{{{d:.2f}}}$')
        plt.legend()
        plt.title(f'Best fit at altitude {waveform_data[0].stats.altitude:.2f} km')
        plt.yscale('log')
     
    #coefs = [coef for coef in coefs] + [dists.min()]
    return fitted(dists_new), coefs, dists.min()

def compute_3d(times, data, vel_seismic, vel_acoustic, r, z):
    
    dt = times[1]-times[0]
    fft_signal = fft(data)
    freq = fftfreq(data.size, dt)
    omega = 2.*np.pi*freq
    sigma = vel_seismic*r
    #print(vel_seismic, r, z)
    if z > 0:
        sigma += vel_acoustic*z
    factor = np.sqrt(abs(omega)/(2.*np.pi*sigma))*np.exp(1j*np.pi*np.sign(omega)/4.)
    print(abs(factor[freq>=0.5][0]))
    return ifft(fft_signal*factor)*signal.windows.tukey(data.size)

def create_one_waveform(folder_waveform, file_waveform, l_stations,
                        no_post_processing=False,
                        target_sampling=82., 
                        max_percentage=0.5, 
                        freq_min=0.1, 
                        freq_max=10,
                        geo_scaling = True,
                        starttime=UTCDateTime.now(),
                        vel_seismic=3.,
                        vel_acoustic=0.34,
                        alpha_geo=0.01,
                        norm_factor_specfem=1e8,
                        remove_time_offset_SPECFEM=False,
                        base_output_folder='OUTPUT_FILES',
                        stf={'type': 'gaussian', 'f0': 1.,}):
    
    waveform_file= f'{folder_waveform}/{base_output_folder}/{file_waveform}'
    waveform_data = pd.read_csv(waveform_file, header=None, delim_whitespace=True)
    waveform_data.columns = ['time', 'data']  
    dt = waveform_data.time.iloc[1]-waveform_data.time.iloc[0]
    
    available_types = ['gaussian']
    if stf:
        #fft_signal = np.fft.fft(waveform_data.data.values)
        if stf['type'] in available_types:
            if stf['type'] == 'gaussian':
                aval = (np.pi*stf['f0'])**2
                alpha = 1.628*stf['f0']
                time_conv = np.arange(-1.5/stf['f0'], 1.5/stf['f0']+dt, dt)
                stf_time = alpha*np.exp(-aval*(time_conv)**2)
                #stf_freq = np.fft.fft(stf_time)
            
            """
            plt.figure()
            plt.plot(waveform_data.time.values, waveform_data.data.values)
            plt.plot(waveform_data.time.values, stf_time)
            plt.figure()
            plt.plot(abs(fft_signal))
            plt.plot(abs(stf_freq))
            """
            convolution = np.convolve(waveform_data.data.values, stf_time, mode='same')*dt
            #plt.figure()
            #plt.plot(convolution, alpha=0.3)
            waveform_data.loc[:,'data'] = convolution
           
    if remove_time_offset_SPECFEM:
        waveform_data = waveform_data.loc[waveform_data.time>0.]
        waveform_data.loc[:,'time'] -= waveform_data.time.min()
    
    network = file_waveform.split('.')[0]
    
    max_character_station = 32
    station_in_df = l_stations.loc[(l_stations.station.str.slice(0,max_character_station)==file_waveform.split('.')[1])&(l_stations.network==network)].iloc[0]
    """
    Geometrical spreading correction is based on an acoustic Green's function assumption.
    See eq. 10 for more details in:
    Simulating three-dimensional seismograms in 2.5-dimensional structures by combining two-dimensional finite difference modelling and ray tracing 
    J. Miksat, T. M. Müller, F. Wenzel, Geophysical Journal International, Volume 174, Issue 1, July 2008, Pages 309–315, 
    https://doi.org/10.1111/j.1365-246X.2008.03800.x
    """
    if geo_scaling and not no_post_processing:
        
        tr = obspy.Trace(); 
        tr.data = waveform_data.data.values; 
        tr.stats.delta = waveform_data.time.iloc[1]-waveform_data.time.iloc[0]; 
        
        times = tr.times()
        data_IS = tr.data.copy()
        data_IS = data_IS*signal.windows.tukey(data_IS.size, alpha=alpha_geo)
        tr.data = compute_3d(times, data_IS, vel_seismic, vel_acoustic, station_in_df.x/1e3, station_in_df.z/1e3)
        waveform_data.loc[:,'data'] = tr.data
    
    dict_tr = dict(
        network = file_waveform.split('.')[0],
        station = file_waveform.split('.')[1],
        channel = file_waveform.split('.')[2],
        starttime = starttime,
        delta = waveform_data.time.iloc[1]-waveform_data.time.iloc[0],
        distance = station_in_df.x/1e3,
        altitude = station_in_df.z/1e3,
        total_path = np.sqrt((station_in_df.x/1e3)**2+(station_in_df.z/1e3)**2),
    )
    tr = obspy.Trace(); 
    tr.data = waveform_data.data.values
    tr.stats.update(dict_tr) 
    if not no_post_processing:
        tr.taper(max_percentage=max_percentage); 
        tr.filter('bandpass', freqmax=freq_max, freqmin=freq_min, zerophase=False)
        tr.resample(target_sampling)
    tr.data *= norm_factor_specfem
    
    return tr
    
def create_waveforms(folder_waveform, file_waveform,
                    remove_stations = [],
                    load_all_waveforms_in_folder=False,
                    folders_combine_mt = {},
                    factors_combine_mt = {'M1': 0.1, 'M2': 0.5, 'M3': 0.8},
                    **kwargs):
    
    ## Check if we need to combine waveforms to build full mt from fundamental ones
    folders_combine_mt_loc = folders_combine_mt.copy()
    combine_waveforms = True
    if not factors_combine_mt:
        folders_combine_mt_loc['M1'] = folder_waveform
        combine_waveforms = False
    else:
        print(f'Combining moment tensors with factors: {factors_combine_mt}')

    ## Loop over fundamental mt simulations or just requested one
    for imt, mt in enumerate(folders_combine_mt_loc):
        waveforms_loc = obspy.Stream()
        folder_waveform_loc = folders_combine_mt_loc[mt]

        station_file = f'{folder_waveform_loc}/DATA/STATIONS'
        l_stations = pd.read_csv(station_file, delim_whitespace=True, header=None)
        l_stations.columns = ['station', 'network', 'x', 'z', 'dummy0', 'dummy1']

        nb_files = 0
        for subdir, dirs, files in os.walk(folder_waveform_loc):
            nb_files += 1
        
        #print(f'Loading directory: {folder_waveform_loc} with {nb_files} directories')
        for subdir, dirs, files in tqdm.tqdm(os.walk(folder_waveform_loc), total=nb_files):
            for file in files:
                #print(f'Loading file: {subdir}/{file}')
                #print(f'Filter: {file_waveform}')
                filepath = subdir + os.sep + file
                if '.semv' in file:
                    skip_this_entry = False
                    for to_remove in remove_stations:
                        if to_remove in file:
                            skip_this_entry = True
                    if ((load_all_waveforms_in_folder) or (file_waveform in file)) and not skip_this_entry:
                        #print(f'Loading file: {subdir}/{file}')
                        waveforms_loc += create_one_waveform(folder_waveform_loc, file, l_stations, **kwargs)
        
        if combine_waveforms:
            factor_mt = factors_combine_mt[mt]
            for tr in waveforms_loc:
                tr.data *= factor_mt
            
        if imt == 0:
            waveforms = waveforms_loc.copy()
        else:
            for tr in waveforms:
                tr_loc = waveforms_loc.select(network=tr.stats.network, station=tr.stats.station)[0]
                tr_loc.trim(endtime=tr.stats.endtime)
                tr.trim(endtime=tr_loc.stats.endtime)
                tr.data += tr_loc.data
            
    print(f'Number of stations loaded: {len(waveforms)}')
                    
    return waveforms