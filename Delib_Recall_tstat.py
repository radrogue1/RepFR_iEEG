'''
last edited by Brandon 2/14

'''

print("loading modules")
import os
from time import time
import csv

import numpy as np 
import pandas as pd
pd.set_option("display.max_columns", 200)

import cmlreaders
from cmlreaders import CMLReader, get_data_index 

import xarray as xr
import scipy.stats as stats
import scipy.spatial as spatial
from sklearn import preprocessing

import mne
from mne import channels
from mne import time_frequency

import h5py


from psifr import fr

import ptsa 
#from ptsa.data.TimeSeriesX import TimeSeries
from ptsa.data.timeseries import TimeSeries
from ptsa.data import timeseries
from ptsa.data.readers import BaseEventReader

from ptsa.data.filters import MorletWaveletFilter
from ptsa.data.filters import ButterworthFilter


import bk_md as bk

print('setting functions')

def save_numpy(array, path, name):
    
    combine = (str(path)+str(name))
    
    np.save( combine , array)
    
    return 


print("setting variables")
sessions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
exp = 'ltpRepFR'
# Run on all sesssions and subjects
ltpRepFR = get_data_index('ltp').query('experiment=="ltpRepFR"')
subjects = get_data_index('ltp').query('experiment=="ltpRepFR"')['subject'].unique()  

ROI_LIST = ['LAS','LAI','LPS','LPI','RAS','RAI','RPS','RPI']
ROIs = {'egi':{'LAS':['E12','E13','E19','E20','E24','E28','E29'],
               'LAI':['E23','E26','E27','E33','E34','E39','E40'],
               'LPS':['E37','E42','E52','E53','E54','E60','E61'],
               'LPI':['E51','E58','E59','E64','E65','E66','E69'],
               'RAS':['E4','E5','E111','E112','E117','E118','E124'],
               'RAI':['E2','E3','E109','E115','E116','E122','E123'],
               'RPS':['E78','E79','E85','E86','E87','E92','E93'],
               'RPI':['E84','E89','E90','E91','E95','E96','E97']},
        'biosemi':{'LAS':['C24','C25','D2','D3','D4','D11','D12','D13'],
                   'LAI':['C31','C32','D5','D6','D9','D10','D21','D22'],
                   'LPS':['D29','A5','A6','A7','A8','A17','A18'],
                   'LPI':['D30','D31','A9','A10','A11','A15','A16'],
                   'RAS':['B30','B31','B32','C2','C3','C4','C11','C12'],
                   'RAI':['B24','B25','B28','B29','C5','C6','C9','C10'],
                   'RPS':['A30','A31','A32','B3','B4','B5','B13'],
                   'RPI':['A28','A29','B6','B7','B8','B11','B12']}}


ROI_order = ['LAS', 'LAI', 'LPS', 'LPI', 'RAS', 'RAI', 'RPS', 'RPI']


        
        
#Define input params 

#Get log-spaced freqs from 3 to 161 Hz
log_freqs = np.array([2**(np.log2(3) + (iFreq/4)) for iFreq in range(24)])
wave_number = 5

#Buffer is one-half the wavelet length
rel_start = -800
rel_stop = 400

buffer_ms = (wave_number * (1000/log_freqs[0])) / 2
sr = 2048.
buffer_samp = int(np.ceil(buffer_ms * (sr/1000.)))
experiment = 'ltpRepFR'

subject_t = []
#subject_z = []
error_log = []

##DELETE THIS AFTER DEBUG 2/17 
#subjects = ['LTP445']

for s in subjects:
    
    select_channels = []

    for x in ROIs['biosemi']:
        for electrode in ROIs['biosemi'][str(x)]:
            
            if electrode[0] != 'A':
                select_channels.append(electrode)
            
            else:
                pass 

    
    print("running "+str(s))
    delib = []
    immediate = []
    
    for sess in sessions: 
        session_d = []
        used_channels = []
        print("running session " + str(sess))
        try:
            n_lists = 26
            events = bk.matched_delibs(s, sess, experiment, n_lists)
            
            events_df = events['delibs']
            
            valid_rec_word = events_df.query("type == 'REC_WORD'")
            matched_silence = events_df.query("type == 'SILENCE_START'")
            all_events = pd.concat([valid_rec_word, matched_silence])
            
            recall_index = np.concatenate([np.repeat(True, valid_rec_word.shape[0]), np.repeat(False, matched_silence.shape[0])])
            delib_index = np.concatenate([np.repeat(False, valid_rec_word.shape[0]), np.repeat(True, matched_silence.shape[0])])
            
            print("events set")
            
            for elec in select_channels:
                try:
                    channel_power = np.load('/scratch/brandon.katerman/RepFR/'+str(s)+'/'+'delib_power/session_'+str(sess)+'/'+str(s)+'_'+str(sess)+'_mt_logRet_Delib_pow_'+str(elec)+'.npy')
                    channel_power = xr.DataArray(channel_power, dims = ['frequency', 'event', 'time'])
                    channel_power = stats.zscore(channel_power, axis = 1)
                    
                    #immediate_power = channel_power[ :, recall_index, :]
                    #delib_power = channel_power[:, delib_index ,: ]
                    #channel_t = stats.ttest_ind(immediate_power, delib_power, equal_var= False, axis = 1).statistic
                    #channel_t = xr.DataArray(channel_t, dims = [ 'frequency', 'time'])
                    
                    session_d.append(channel_power)
                    del channel_power

                    used_channels.append(elec)

                except Exception as e:
                    print(e)
                    error_log.append(e)
            if len(used_channels) != 0: 
                select_channels = used_channels
        except Exception as e:
            print('error session '+str(sess))
            error_log.append(e)
            print(e)

        if len(session_d) > 0:
            try:
                session_d = xr.DataArray(session_d, dims = ['channels', 'freqs', 'event', 'timepoint'])
                #session_t = xr.concat(session_t, dim = 'channels')
                session_d.coords['channels'] = np.asarray(used_channels)
                #z_pows = stats.zscore(session_powers, axis = 2)
                


                immediate.append(session_d[:, :, recall_index , :])
                delib.append(session_d[:, :, delib_index, :])
                #del z_pows
            except Exception as e:
                print(e)
                error_log.append(e)
        else:
            pass 
    
    all_de = np.concatenate(delib, axis = 2)
    all_im = np.concatenate(immediate, axis = 2)
    tvals = stats.ttest_ind( all_im,all_de, equal_var=False, axis = 2).statistic
    x_tvals = xr.DataArray(tvals, dims = ['channels','freqs','timepoints'],
                      coords = {'channels': select_channels})
    #split_t = split_ROIs(x_tvals, 'biosemi')
    
    #change z save to be subject level and not all at once 
    subject_z = [all_de, all_im]
    
    #save_numpy(subject_z, '/scratch/brandon.katerman/RepFR/','{}_delib_imrec_z_pows_mdDebug217'.format(s))
    save_numpy(subject_z, '/scratch/brandon.katerman/RepFR/','{}_delib_imrec_z_pows_NO_A'.format(s))
    print("saved")

    subject_t.append(x_tvals)
 


    
#save_numpy(subject_t, '/scratch/brandon.katerman/RepFR/', 'all_delib_imrec_t_stat_mdDebug217')
save_numpy(subject_t, '/scratch/brandon.katerman/RepFR/', 'all_delib_imrec_t_stats_NO_A')
#save_numpy(subject_z, '/scratch/brandon.katerman/RepFR/','all_delib_imrec_z_pows_NO_A')

error_log_path = "/scratch/brandon.katerman/RepFR/"


# textfile = open(error_log_path+"Delib_Recall_error_log.txt", 'w')
# for error in error_log:
#     textfile.write(error + '\n')
# textfile.close()

print("DONE :)")