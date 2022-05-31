'''
Created 2/24 by Brandon Katerman * 


'''


import os
from time import time
import csv
import sys

import numpy as np 
import pandas as pd
pd.set_option("display.max_columns", 200)
import cmlreaders as cml
from cmlreaders import CMLReader, get_data_index 
import xarray as xr
import scipy.stats as stats
import scipy.spatial as spatial
from sklearn import preprocessing

import mne
from mne import channels
from mne import time_frequency

import h5py

import glob

import ptsa 
#from ptsa.data.TimeSeriesX import TimeSeries
from ptsa.data.timeseries import TimeSeries
from ptsa.data import timeseries
from ptsa.data.readers import BaseEventReader

from ptsa.data.filters import MorletWaveletFilter
from ptsa.data.filters import ButterworthFilter

import matched_delib as md

def save_numpy(array, path, name):

    combine = (str(path)+str(name))

    np.save( combine , array)

    return 

def Create_Epoch(eeg, buffer_in_seconds, events, start, stop):

    '''
    convert raw eeg to an MNE epoch and then convert to a TimeSeries object for better data manipulation 

    runs average reference and resamples data 

    '''
    def add_mirror_buffer(self, duration):
        """
        Return a time series with mirrored data added to both ends of this
        time series (up to specified length/duration).
        The new series total time duration is:
            ``original duration + 2 * duration * samplerate``
        Parameters
        ----------
        duration : float
            Buffer duration in seconds.
        Returns
        -------
        New time series with added mirrored buffer.
        """
        samplerate = float(self['samplerate'])
        samples = int(np.ceil(float(self['samplerate']) * duration))
        if samples > len(self['time']):
            raise ValueError("Requested buffer time is longer than the data")

        data = self.data

        mirrored_data = np.concatenate(
            ( data,
             data[..., -samples - 1:-1][..., ::-1]), axis=-1)

        start_time = self['time'].data[0] - duration
        t_axis = (np.arange(mirrored_data.shape[-1]) *
                  (1.0 / samplerate)) + start_time
        # coords = [self.coords[dim_name] for dim_name in self.dims[:-1]] +[t_axis]
        coords = {dim_name:self.coords[dim_name]
                  for dim_name in self.dims[:-1]}
        coords['time'] = t_axis
        coords['samplerate'] = float(self['samplerate'])

        return TimeSeries(mirrored_data, dims=self.dims, coords=coords)
    
    mne_events = np.zeros((len(events), 3), dtype = int)
    mne_events[:,0] = [o for i, o in enumerate(events['eegoffset'])]
#     print("maybe here?")
    ep = mne.Epochs(eeg, mne_events, tmin =  start, tmax = stop, baseline = None, preload = True, on_missing = 'ignore', verbose = False)
    ep._data = ep._data * 1000000
    ep.filter(l_freq = 0.5, h_freq = None, method = 'iir', iir_params = None)
    ep.pick_types(eeg= True, exclude = [])
    #do  i need this part? maybe not
    #ep.resample(500.0)
    '''Create an xarray version of epoch data'''
    x = TimeSeries(ep._data, dims=('events','channels','time'),
                              coords={'events':events.to_records(),
                                      'channels':ep.info['ch_names'],
                                      'time':ep.times,
                                      'samplerate':ep.info['sfreq']})
    print(" and mirroring ...", end = '')
    x = add_mirror_buffer(x, buffer_in_seconds)

    print("<3")

    return ep, x, mne_events


def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

            
def Channel_Powers(subject, sess, channels, eeg, buffer_samp, freqs, wave_number, path, log = True):
    for name, i in zip(channels, range(len(channels))):
        channel_eeg = eeg[:, i, :]
        wave_pows = MorletWaveletFilter(freqs = freqs, width = wave_number, output = 'power').filter(timeseries = channel_eeg)
        wave_pows = wave_pows.data.astype(np.float32)
        i_wave_pows = wave_pows[:, :, buffer_samp:-buffer_samp]
        if log == True:
            i_wave_pows = np.log10(i_wave_pows)
        else:
            pass 
        print(str(name)+" saved.")
        folder = path.format(subject, sess)
        create_folder(folder)
        save_numpy(i_wave_pows,folder, "{}_{}_mt_logRet_Delib_{}_pow".format(subject ,sess, name))

def get_rec_powers(subject, pairs, hemisphere, region):
    freqs = np.array([2**(np.log2(3) + (iFreq/4)) for iFreq in range(24)])
    n_lists = 21
    experiment='RepFR1'
    wave_number = 5
    half_wav = ((1000 / freqs.min()) * wave_number) * 0.5
    index = get_data_index('r1'); index=index[(index.subject==subject) & (index.experiment=='RepFR1')]
    sessions = index.session.to_numpy()
    results = []
    for sess in sessions: 
        rec_win = [-750, 400]
        recall_start = rec_win[0] / 1000
        recall_stop = rec_win[1] / 1000
        
        rec_dif = np.diff([recall_start, recall_stop])

        silence_start = 0 
        silence_stop = rec_dif[0]

        print("running session {}".format(sess))
        try:
            r = cml.CMLReader(subject=subject, experiment='RepFR1', session=sess)
            eeg = r.load_eeg(scheme=pairs)
            channels = eeg.channels

            sr = float(eeg.samplerate)

            events = md.matched_delibs(subject, 
                                       sess, 
                                      experiment, 
                                       n_lists, 
                                       freqs, 
                                       wave_number, 
                                       rec_win, 
                                       sr, 
                                       pre_voc_exclusion = (abs(rec_win[0])), 
                                       voca_dur = 550)

            valid_rec_word = events['delibs'].query("type == 'REC_WORD'")
            matched_silence = events['delibs'].query("type == 'SILENCE_START'")
            all_events = pd.concat([valid_rec_word, matched_silence])
            print("\n")
            print("pulling eeg")

            buffer_ms = (wave_number * (1000/freqs[0])) / 2
            buffer_samp = int(np.ceil(buffer_ms * (sr/1000.)))
            buffer_minimized = buffer_ms/1000
            if hemisphere == '':
                path = '/scratch/radrogue/RepFR1/{}/{}/matched_delib/session_{}/'.format(region, subject, sess)
            else:
                path = '/scratch/radrogue/RepFR1/{}/{}/matched_delib/session_{}/'.format(hemisphere+'_'+region, subject, sess)

            print("\n")
            print("converting to epoch")

            rec_epoch, rec_x, rec_mne_events = Create_Epoch(eeg.to_mne(), buffer_minimized, valid_rec_word, 
                                                            recall_start+(-1*buffer_minimized), recall_stop)

            sil_epoch, sil_x, sil_mne_events = Create_Epoch(eeg.to_mne(), buffer_minimized, matched_silence, 
                                                            silence_start+(-1*buffer_minimized), silence_stop)

            rec_eeg = ButterworthFilter(freq_range =[58., 62.], filt_type = 'stop', order = 4).filter(timeseries=rec_x)
            sil_eeg = ButterworthFilter(freq_range =[58., 62.], filt_type = 'stop', order = 4).filter(timeseries=sil_x)

            channels = rec_eeg.channels 
            all_eeg = xr.DataArray(np.concatenate([rec_eeg, sil_eeg], 0), dims = ['events', 'channels', 'timepoint'],
                                  coords = {'events':all_events.to_records(),
                                      'channels': rec_eeg.channels})

            eeg = TimeSeries(all_eeg, dims = ('events', 'channels', 'time'),
                           coords = {'events': all_eeg.events,
                                    'channels': all_eeg.channels,
                                    'samplerate': sr })
            Channel_Powers(subject, sess, channels, eeg, buffer_samp, freqs, wave_number, path)
            results.append(subject+'_'+str(sess) +' worked :)')
        except Exception as e:
            results.append(subject+'_'+str(sess) +' failed :( ' + str(e))
    return results

def rec_power_statistics(subject, pairs, hemisphere, region):
    freqs = np.array([2**(np.log2(3) + (iFreq/4)) for iFreq in range(24)])
    n_lists = 21
    experiment='RepFR1'
    wave_number = 5
    half_wav = ((1000 / freqs.min()) * wave_number) * 0.5
    rec_win = [-750, 400]
    recall_start = rec_win[0] / 1000
    recall_stop = rec_win[1] / 1000
    rec_dif = np.diff([recall_start, recall_stop])

    silence_start = 0 
    silence_stop = rec_dif[0]
    delib = []
    immediate = []
    bad = []
    if hemisphere == '':
        subj_path = '/scratch/radrogue/RepFR1/{}/{}/matched_delib/'.format(region,subject)
    else:
        subj_path = '/scratch/radrogue/RepFR1/{}/{}/matched_delib/'.format(hemisphere+'_'+region,subject)
        
    sessions = []
    for root, dirs, files in os.walk(subj_path):
        for dire in dirs:
            sessions.append(int(dire[-1]))
    for sess in sessions:
        if hemisphere == '':
            path = '/scratch/radrogue/RepFR1/{}/{}/matched_delib/session_{}/'.format(region,subject,sess)
        else:
            path = '/scratch/radrogue/RepFR1/{}/{}/matched_delib/session_{}/'.format(hemisphere+'_'+region,subject,sess)
        print("running session {}".format(sess), path)
        try:
            print("running session {}".format(sess))
            r = cml.CMLReader(subject=subject, experiment='RepFR1', session=sess)
            eeg = r.load_eeg(scheme=pairs)
            channels = eeg.channels

            sr = float(eeg.samplerate)

            events = md.matched_delibs(subject, 
                                       sess, 
                                      experiment, 
                                       n_lists, 
                                       freqs, 
                                       wave_number, 
                                       rec_win, 
                                       sr, 
                                       pre_voc_exclusion = (abs(rec_win[0])), 
                                       voca_dur = 550)
            events_df = events['delibs']
            
            valid_rec_word = events_df.query("type == 'REC_WORD'")
            matched_silence = events_df.query("type == 'SILENCE_START'")
            all_events = pd.concat([valid_rec_word, matched_silence])
            
            recall_index = np.concatenate([np.repeat(True, valid_rec_word.shape[0]), np.repeat(False, matched_silence.shape[0])])
            delib_index = np.concatenate([np.repeat(False, valid_rec_word.shape[0]), np.repeat(True, matched_silence.shape[0])])
            

            print("done.")

            print("pulling channel power")
            session_power = []
            channel_power = []
            for root, dire, files in os.walk(path):
                for file in files:
                    data = np.load(root+file)
                    channel_power = xr.DataArray(data, dims = ['frequency', 'event', 'time'])
                    channel_power_z = stats.zscore(channel_power, axis = 1, ddof = 1)
                    session_power.append(channel_power_z); del channel_power_z; del channel_power; del data;

            session_d = xr.DataArray(session_power, dims = ['channels', 'freqs', 'event', 'time'])
            session_d.coords['channels'] = channels
            print("splitting events")
            immediate.append(session_d[:, :, recall_index , :])
            delib.append(session_d[:, :, delib_index, :])
        except Exception as e:
            print(subject, sess, 'failed:', e)
            bad.append(subj+"_"+str(sess) + ' failed: '+ str(e))
    try:
        all_de = np.concatenate(delib, axis = 2)
        all_im = np.concatenate(immediate, axis = 2)
        tvals = stats.ttest_ind( all_im,all_de, equal_var=False, axis = 2).statistic
        x_tvals = xr.DataArray(tvals, dims = ['channels','freqs','timepoints'],
                          coords = {'channels': channels})
        #split_t = split_ROIs(x_tvals, 'biosemi')

        #change z save to be subject level and not all at once 
        subject_z = [all_de, all_im]

        #save_numpy(subject_z, '/scratch/brandon.katerman/RepFR/','{}_delib_imrec_z_pows_mdDebug217'.format(s))
        save_numpy(subject_z, subj_path,'{}_delib_imrec_z_pows_NO_A'.format(subject))
        save_numpy(x_tvals, subj_path,'{}_delib_imrec_t_pows_NO_A'.format(subject))
        print("saved")
        print("DONE :)")
        return (subject + ' <3')
    except Exception as e:
        return (subject + ' failed :( ' + str(e))