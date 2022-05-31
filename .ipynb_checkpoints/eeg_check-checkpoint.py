import numpy as np
from ptsa.data.filters import ResampleFilter, ButterworthFilter, MorletWaveletFilter
import cmlreaders as cml
import xarray as xr
import cc_utils as cc
from scipy import fft
from scipy.stats import linregress
from scipy.stats import zscore
import pandas as pd
from sys import exc_info, getsizeof

def filter_and_resample(ch_data, FREQS, POWHZ, buf):
    ch_data = ButterworthFilter(freq_range=[58, 62], filt_type='stop', order=4).filter(timeseries=ch_data)
    ch_data = MorletWaveletFilter(freqs=FREQS, output='power',width=5, verbose=True).filter(timeseries=ch_data)
    ch_data = ch_data.remove_buffer(duration=(buf / 1000))
    ch_data = xr.ufuncs.log10(ch_data, out=ch_data.values)
    ch_data = ch_data.mean('time')
    return ch_data
    
def eeg_check(subject, session, exp):
    period = 'encoding' # 'retrieval' 'encoding'
    POWHZ = 50. # downsampling factor
    FREQS = np.logspace(np.log10(3),np.log10(180), 8)

    df = cml.CMLReader.get_data_index()
    exp_df = df[(df.subject == subject) & (df.experiment == exp) & (df.session == session)]
    del df;

    if len(exp_df) != 1:
        raise KeyError('The wrong number of sessions inputted' + str(l))
    pow_params = {  
        'RepFR1': {
            'encoding': {
                'mirror': False,
                'start_time': 0.3,
                'end_time': 1.3,
                'buf': 1.0, #2.0
                'ev_type': ['WORD']
            },
            'retrieval': {
                'mirror': True,
                'start_time': -0.9,
                'end_time': -0.1,
                'buf': 0.76,
                'ev_type': ['REC_WORD', 'REC_BASE']
            },
            'countdown': {
                'mirror': False,
                'start_time': 1,
                'end_time': 2,
                'buf': 1,
                'ev_type': ['COUNTDOWN']
            }           
        },
    }
    reader = cml.CMLReader(subject=exp_df['subject'].iloc[0], 
                           experiment=exp_df['experiment'].iloc[0], session=exp_df['session'].iloc[0], 
                           localization=exp_df['localization'].iloc[0], montage=exp_df['montage'].iloc[0])
    del exp_df;
    events = reader.load('task_events')
    ev_type = pow_params[exp][period]['ev_type']
    events = events.query('type == @ev_type')
    del ev_type; 
    if period == 'encoding':
        events = cc.process_events(events)
    elif period == 'retrieval':
        events = cc.process_events_retrieval(events) 
    elif period == 'countdown':
        events = cc.process_events_countdown(events) 

    scheme = reader.load("pairs")
    buf = pow_params[exp][period]['buf'] * 1000
    rel_start = pow_params[exp][period]['start_time'] * 1000
    rel_stop = pow_params[exp][period]['end_time'] * 1000

    if pow_params[exp][period]['mirror']: # use mirror for retrieval for unknown reasons
        dat = reader.load_eeg(events=events,
                          rel_start=rel_start,
                          rel_stop=rel_stop,
                          scheme=scheme).to_ptsa()
        dat['time'] = dat['time'] / 1000 # using PTSA time scale
        dat = dat.add_mirror_buffer(pow_params[exp][period]['buf'])
        dat['time'] = dat['time'] * 1000
    else:
        dat = reader.load_eeg(events=events,
                          rel_start=-buf+rel_start,
                          rel_stop=buf+rel_stop,
                          scheme=scheme).to_ptsa()
    del reader; del events; del pow_params; del exp;
    del rel_start; del rel_stop;

    dat = dat.astype(float) - dat.mean('time')
    all_data = []
    for ch in np.arange(dat.shape[1]):
        ch_data = dat[:,ch,:]
        ch_data = filter_and_resample(ch_data=ch_data, FREQS=FREQS, POWHZ=POWHZ, buf=buf)
        all_data.append(ch_data)
    # do each step twice with split channels
#     half_elec = int(np.shape(dat)[1]/2)
    # Notch Butterworth filter for 60Hz line noise:
#     dat1 = ButterworthFilter(freq_range=[58, 62], filt_type='stop', order=4).filter(timeseries=dat[:,:half_elec,:])
#     dat2 = ButterworthFilter(freq_range=[58, 62], filt_type='stop', order=4).filter(timeseries=dat[:,half_elec:,:])
#     dat1 = MorletWaveletFilter(freqs=FREQS, output='power',width=5, verbose=True).filter(timeseries=dat1)
#     dat2 = MorletWaveletFilter(freqs=FREQS, output='power',width=5, verbose=True).filter(timeseries=dat2)
#     del FREQS;
#     dat1 = xr.ufuncs.log10(dat1, out=dat1.values)
#     dat2 = xr.ufuncs.log10(dat2, out=dat2.values)

#     dat1 = ResampleFilter(resamplerate=POWHZ).filter(timeseries=dat1) # 8 x trials x elecs x samples resampled to 20 ms bins
#     dat2 = ResampleFilter(resamplerate=POWHZ).filter(timeseries=dat2)
#     del POWHZ; 
    #     print('done resample')

    filter_noise_lines = 1

    if filter_noise_lines == 1:
        # line noise removal
        line_filt_dat = ButterworthFilter(freq_range=[58., 62.], filt_type='stop', order=4).filter(timeseries=dat)
        line_filt_dat = ButterworthFilter(freq_range=[118., 122.], filt_type='stop', order=4).filter(timeseries=line_filt_dat)        
        line_filt_dat = ButterworthFilter(freq_range=[178., 182.], filt_type='stop', order=4).filter(timeseries=line_filt_dat)                
    else:
        line_filt_dat = dat     
    del dat;
    print(line_filt_dat)
    dat = xr.DataArray(all_data, dims = ['channel', 'frequency', 'events'])
    print(dat.shape)
    del all_data;
#     dat = dat.mean('time') # average over time so 8 x trials x elecs

    if (period == 'encoding') | (period == 'countdown'):
        z_pow_all = xr.apply_ufunc(zscore, dat, 
                                   input_core_dims=[['channel', 'frequency']], 
                                   output_core_dims=[['channel', 'frequency']]) # outputs trials x elecs x 8
    elif period == 'retrieval':
        z_pow_all = xr.apply_ufunc(zscore, dat, 
                                   input_core_dims=[['channel', 'frequency', 'time']], 
                                   output_core_dims=[['channel', 'frequency', 'time']])
    del dat; del period;
    print(z_pow_all)
    N = 1000 #np.shape(line_filt_dat)[2] # Number of samplepoints 
    T = 1.0 / N # sample spacing 
    sr = line_filt_dat.samplerate.values

    time_range = range(0,int(2000*(sr/1000))) # 2000 ms
    plot_range = range(5,382) # first few are strange...then only go to 200 Hz (which is 400)
    shift_factor = 1 # taking different time chunks shifts the frequencies
    ch_resid = []
    ch_z_pow = []
    for ch in range(np.shape(line_filt_dat)[1]):
        abs_resid = []
        z_pows = []
        for tr in range(np.shape(line_filt_dat)[0]):
            y = line_filt_dat[tr,ch,time_range]
            yf = fft(y) 
            # get the frequency spectrum after the fft by removing mirrored signal and taking modulus
            corrected_yf = shift_factor/N * np.abs(yf[:N//2]) 
            xf = np.linspace(0.0, N/2, N)    
            xplot = xf[plot_range]
            yplot = corrected_yf[plot_range]
            slope, intercept, r_value, _, _ = linregress(np.log(xplot), np.log(yplot))
            spectrum_val = np.polyval([slope,intercept],np.log(xplot))
            abs_resid.append(np.mean(np.abs([np.exp(n) for n in spectrum_val]-np.log(yplot))))
            if tr < np.shape(line_filt_dat)[0]-1: # last one don't look for correlation
                z_pows.append(np.corrcoef(z_pow_all[tr,ch,:],z_pow_all[tr+1,ch,:])[0,1])
        ch_z_pow.append(np.mean(z_pows))
        ch_resid.append(np.mean(abs_resid))

    scheme['bad'] = pd.Series(0, scheme.index)
    data1 = ch_z_pow
    data2 = ch_resid

    two_std1 = np.mean(data1)+2*np.std(data1)
    two_std2 = np.mean(data2)+2*np.std(data2)

    for index1, datum1 in enumerate(data1):
        if datum1 >= two_std1:
            scheme.at[index1, ['bad']] = 1

    for index2, datum2 in enumerate(data2):
        if datum2 >= two_std2:
            scheme.at[index2, ['bad']] = 1
    scheme['subject'] = pd.Series(subject, scheme.index)
    scheme['session'] = pd.Series(session, scheme.index)
    return scheme