print("loading modules")
import os
from time import time
import csv
import cmlreaders as cml
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

import glob

import ptsa 
#from ptsa.data.TimeSeriesX import TimeSeries
from ptsa.data.timeseries import TimeSeries
from ptsa.data import timeseries
from ptsa.data.readers import BaseEventReader

from ptsa.data.filters import MorletWaveletFilter
from ptsa.data.filters import ButterworthFilter

def filter_repetitions(events, list_index):

    '''
    two functions that split recall events that contain repeated word presentations into a new events dataframe containing only 
    the first and second presentation of repeated words. Then additionally outputs two masks to additionally filter the already filtered 
    recalls by either their first and second presentation.

    This is to ideally split first/second presentations by recalled and not-recalled to calculate two different SMEs

    Parameters
    ----------

    events: encoding events ideally from ltpRepFR and a similar repeated word presentation experiment 

    list_index: list of all list values (in ltpepFr 1 through 25)

    Return
    -------

    filtered_events: events that have one time presented words and the third presentation of repeated words removed 

    first: array mask to only include the first presentation of filtered repeated encoding events 

    second: array mask to only include the second presentation of filtered repeated encoding events 

    '''

    def pull_first_and_second_rep_presentations(events, list_index):

        '''
        take in events and list_index structure from ltpRepFR data, filter events to only include first and second presentation of 
        second and three times repeated words -- excluding practice list (list 0) and all single presentation words 


        Parameters
        ----------

        events: ltpRepFR word events 

        list_index: list of all list values (in ltpRepFr 1 through 25)

        Return
        -------
        event_mask: array of True and False values for all events -- True events are those that are 2 or 3 times repeated words and only their first and second presentations 
        '''

        session_array = []
        session_array.extend(np.repeat(False, 27))

        for l in list_index:
            current_list = events[events['list'] == l] #interate one list at a time 
            non_repeat = list(current_list[current_list['repeats'] == 1].item_name)

            #dict to track number of presentations of a word during one list 
            pres_dict = {}
            for x in np.unique(current_list.item_name):
                pres_dict['{0}'.format(x)] = 0

            for word in current_list.item_name:

                #if a once repeated word -- append False value and increment that word by 1 presentation 
                if word in non_repeat:
                    session_array.append(False)
                    pres_dict[str(word)] += 1 

                elif word not in non_repeat:
                    #presentation value 
                    presentation = pres_dict[str(word)]

                    #if word has reached or is reaching second presentation -- append True value and increment that word by 1 presentation 
                    if presentation <= 2:
                        session_array.append(True) 
                        pres_dict[str(word)] += 1 

                    elif presentation > 2:
                        session_array.append(False)
                        pres_dict[str(word)] += 1 

        event_mask = np.asarray(session_array).flatten()

        if event_mask.shape[0] != 567: 
            print("missing "+str(567 - int(np.asarray(session_array).shape[0]))+' encoding events. Not Complete.')

        return event_mask



    def split_first_and_second_pres_events(events, list_index):
        '''
        After removing single presentation words and the third presentation of repeated words, 
        specifically split the results encoding event array by first and second presentations and create respective masks 


        Parameters
        -----------
        events: filtered events from "pull_first_and_second_rep_presentation function"

        list_index: list of all list values (in ltpRepFr 1 through 25)

        Return
        -------

        first_p_mask: mask to filter event array for only the first presentations of repeated words 

        second_p_masl: mask to filter event array for only the second presentations of repeated words 
        '''

        first_p = []
        second_p = []
        for l in list_index:
            current_list = events[events['list'] == l]

            pres_dict = {}
            for x in np.unique(current_list.item_name):
                pres_dict['{0}'.format(x)] = 0

            for word in current_list.item_name:
                presentation = pres_dict[str(word)]

                if presentation == 0:
                    first_p.append(True)
                    second_p.append(False)
                    pres_dict[str(word)] += 1 

                elif presentation == 1:
                    first_p.append(False)
                    second_p.append(True)
                    pres_dict[str(word)] += 1 


        first_p_mask = np.asarray(first_p)
        second_p_mask = np.asarray(second_p)


        return first_p_mask, second_p_mask 

    event_mask = pull_first_and_second_rep_presentations(events, list_index)
    print(len(events))
    filtered_events = events[event_mask]
    first, second = split_first_and_second_pres_events(filtered_events, list_index)

    return filtered_events, first, second 


def save_numpy(array, path, name):

        combine = (str(path)+str(name))

        np.save( combine , array)

        return 


def Create_Epoch(eeg, events, start, stop):
    '''
    convert raw eeg to an MNE epoch and then convert to a TimeSeries object for better data manipulation
    runs average reference and resamples data
    '''
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
    return ep, x, mne_events





def Channel_Powers(subject, session, channels, eeg, buffer_samp, wave_number, freqs, path, error_log_path, log = True):

    channels = channels

    def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    for name, i in zip(channels, range(len(channels))):
        channel_eeg = eeg[:, i, :]
#         print(channel_eeg)
#         wave_pows = MorletWaveletFilter(channel_eeg, freqs = freqs, width = wave_number, output = 'power').filter()
        wave_pows = MorletWaveletFilter(freqs = freqs, width = wave_number, output = 'power').filter(timeseries = channel_eeg)
        wave_pows = wave_pows.data.astype(np.float32)
        i_wave_pows = wave_pows[:, :, buffer_samp:-buffer_samp]
        if log == True:
            i_wave_pows = np.log10(i_wave_pows)
        else:
            pass 

        print(str(name)+" saved.")
        
        folder = path.format(subject , session)
        create_folder(folder)
        save_numpy(i_wave_pows,folder, "{}_{}_mt_logRet_enc_pow_{}".format(subject ,session, name))


def get_enc_powers(subject, pairs, hemisphere, region):
    results = []
    data = cml.get_data_index(kind = 'r1'); data = data[data['experiment'] == 'RepFR1']
    sessions = data[data.subject == subject].session.unique()
    freqs = np.array([2**(np.log2(3) + (iFreq/4)) for iFreq in range(24)])
    rel_start = 0/1000
    rel_stop = 1600/1000
    wave_number = 5
    half_wav = ((1000 / freqs.min()) * wave_number) * 0.5
    for sess in sessions:
        try:
            print(subject, sess, 'loading...')
            r = cml.CMLReader(subject=subject, experiment='RepFR1', session=sess)
            evs = r.load('task_events')
            word_evs = evs.query("type == 'WORD'")
            word_evs = word_evs[word_evs.list != -999]
            word_evs = word_evs[word_evs.list!=0]            
            print(subject, sess, 'filtering...')
            eeg = r.load_eeg(scheme=pairs)
            channels = eeg.channels

            
            sr = float(eeg.samplerate)

            buffer_ms = (wave_number * (1000/freqs[0])) / 2
            buffer_samp = int(np.ceil(buffer_ms * (sr/1000.)))
            buffer_minimized = buffer_ms/1000
            if hemisphere == '':
                path = '/scratch/radrogue/RepFR1/{}/{}/sme_pow_debug/session_{}/'.format(region, subject, sess)
            else:
                path = '/scratch/radrogue/RepFR1/{}/{}/sme_pow_debug/session_{}/'.format(hemisphere+'_'+region, subject, sess)
            ep, enc_x, mne_evs = Create_Epoch(eeg.to_mne(), word_evs, rel_start + (-1 * buffer_minimized), rel_stop + buffer_minimized)
            if np.all(enc_x == 0):
                raise Exception('No eeg data for this session')
            enc_eeg = ButterworthFilter(freq_range =[58., 62.], filt_type = 'stop', order = 4).filter(enc_x)
            print(subject, sess, 'calculating powers...')
            Channel_Powers(subject = subject, session = sess, channels = channels, eeg=enc_eeg, buffer_samp=buffer_samp, wave_number = wave_number, freqs=freqs, path=path, error_log_path = '', log = True)
            results.append(subject+'_'+str(sess) + ' worked!')
        except Exception as e:
            print(e)
            results.append(subject+'_'+str(sess) + ' failed: '+str(e))
    return results


def enc_power_statistics(subject, pairs, hemisphere, region):
    subj = subject
    exp = 'RepFR1'
    
    subj_first = []
    subj_first_n = []
    subj_first_r  = []
    
    subj_second = []
    subj_second_n = []
    subj_second_r = []

    subj_three = []
    subj_three_n = []
    subj_three_r = []
    
    index = cml.get_data_index('r1')

    COND = 'repeat'
    sessions =[]
    bad = []
    if hemisphere == '':
        subj_path = '/scratch/radrogue/RepFR1/{}/{}/sme_pow_debug/'.format(region,subject)
    else:
        subj_path = '/scratch/radrogue/RepFR1/{}/{}/sme_pow_debug/'.format(hemisphere+'_'+region,subject)
    for root, dirs, files in os.walk(subj_path):
        for dire in dirs:
            sessions.append(int(dire[-1]))
    print(sessions)
    for sess in sessions:
        if hemisphere == '':
            path = '/scratch/radrogue/RepFR1/{}/{}/sme_pow_debug/session_{}/'.format(region,subject,sess)
        else:
            path = '/scratch/radrogue/RepFR1/{}/{}/sme_pow_debug/session_{}/'.format(hemisphere+'_'+region,subject,sess)
        print("running session {}".format(sess), path)
        try:
            print("filtering events .. ", end = '')
            r = cml.CMLReader(subj, exp, sess)
            events = r.load("task_events")
            word_evs = events.query("type == 'WORD'")
            word_evs = word_evs[word_evs.list != -999]
            word_evs = word_evs[word_evs.list!=0]
            word_evs = word_evs.reset_index()
            list_index = word_evs.list.unique()[1:]
            select_channels = pairs.label.unique()
            no_rep_evs = word_evs[(word_evs.repeats==1) & (word_evs.list != 0)]

            present1_recall = no_rep_evs.recalled.to_numpy()
            present1_forgotten = ~present1_recall

            f_evs = word_evs[(word_evs.repeats>1) & (word_evs.list != 0)]
            first_filter = f_evs.drop_duplicates(subset = 'item_name', keep = 'first')
            second_filter = f_evs.loc[~f_evs.index.isin(first_filter.index)].drop_duplicates(subset = 'item_name', keep = 'first')

            three_times = f_evs[f_evs.repeats == 3]
            print(three_times)
            three_recalled = three_times.recalled.to_numpy()
            three_forgotten = ~three_recalled
    #             f_evs = pd.concat([first, second]).reset_index().drop('index', 1)

    #             first_filter = f_evs.drop_duplicates(subset = 'item_name', keep = 'first')
    #             second_filter = f_evs.loc[~f_evs.index.isin(first_filter.index)]
            #repeated words first presentation 
            first_recalled = first_filter.recalled.to_numpy()
            first_forgotten = ~first_recalled

            #repeated words second presentation 
            second_recalled = second_filter.recalled.to_numpy()            
            second_forgotten = ~second_recalled 

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
            session_d.coords['channels'] = np.asarray(select_channels)
            print("splitting events")
            first_presentation = session_d[:, :, first_filter.index, :]; subj_first.append(first_presentation);
            first_forgot = first_presentation[:, :, first_forgotten, :]; subj_first_n.append(first_forgot)
            first_rec = first_presentation[:, :, first_recalled, :]; subj_first_r.append(first_rec)
            del first_presentation; del first_forgot; del first_rec;


            second_presentation = session_d[:, :, second_filter.index, :]; subj_second.append(second_presentation)
            second_forgot = second_presentation[:, :, second_forgotten, :]; subj_second_n.append(second_forgot)
            second_rec = second_presentation[:, :, second_recalled, :]; subj_second_r.append(second_rec)
            del second_presentation; del second_forgot; del second_rec;

            three_presentation = session_d[:, :, three_times.index, :]; subj_three.append(three_presentation);
            three_forgot = three_presentation[:, :, three_forgotten, :]; subj_three_n.append(three_forgot); del three_forgot;
            three_remembered = three_presentation[:, :, three_recalled, :]; subj_three_r.append(three_remembered); del three_remembered
            del three_presentation;

            del session_d

        except Exception as e:
            print(subj, sess, 'failed:', e)
            bad.append(subj+"_"+str(sess) + ' failed: '+ str(e))
    try:
        print('concatenating subject data')
        session_firstN = np.concatenate(subj_first_n, axis = 2)
        del subj_first_n;
        session_firstR = np.concatenate(subj_first_r, axis = 2)
        del subj_first_r;
        session_secondN = np.concatenate(subj_second_n, axis = 2)
        del subj_second_n;
        session_secondR = np.concatenate(subj_second_r, axis =2)
        del subj_second_r
        session_first = np.concatenate(subj_first, axis = 2)
        del subj_first;
        session_second = np.concatenate(subj_second, axis = 2)
        del subj_second;
        print("running stats")
        session_threeN = np.concatenate(subj_three_n, axis = 2)
        del subj_three_n;
        session_threeR = np.concatenate(subj_three_r, axis = 2)
        del subj_three_r;
        session_three = np.concatenate(subj_three, axis = 2)
        del subj_three;
        
        # *** First Presentation Remembered - Forgotten *** #
        first_tvals = stats.ttest_ind(session_firstR, session_firstN, equal_var = False, axis = 2).statistic

        check_first_tvals = abs(first_tvals)
        if check_first_tvals[check_first_tvals>7].shape[0] > 0:
            raise Exception('A channel for first presentation has a t-value over 7! Bad.')

        first_x = xr.DataArray(first_tvals, dims = ['channels', 'freqs', 'timepoints'], 
                          coords = {'channels': select_channels})

        # *** Second Presentation Remembered - Forgotten *** #
        second_tvals = stats.ttest_ind(session_secondR, session_secondN, equal_var = False, axis = 2).statistic

        check_first_tvals = abs(first_tvals)
        if check_first_tvals[check_first_tvals>7].shape[0] > 0:
            raise Exception('A channel for first presentation has a t-value over 7! Bad.')
        second_x = xr.DataArray(second_tvals, dims = ['channels', 'freqs', 'timepoints'], 
                              coords = {'channels': select_channels})

        # *** Presentations 2 - 1 *** #
        all_pres_ttest = stats.ttest_rel(session_second, session_first, axis = 2).statistic

        check_all_pres_ttest = abs(all_pres_ttest)
        if check_all_pres_ttest[check_all_pres_ttest>7].shape[0] > 0:
            raise Exception('A channel for first presentation has a t-value over 7! Bad.')

        all_pres_x = xr.DataArray(all_pres_ttest, dims = ['channels', 'freqs', 'timepoints'], 
                              coords = {'channels': select_channels})

        # *** Second Remembered - First Remembered *** #
        all_R_ttest = stats.ttest_rel(session_secondR, session_firstR, axis = 2).statistic

        check_all_R_ttest = abs(all_R_ttest)
        if check_all_R_ttest[check_all_R_ttest>7].shape[0] > 0:
            raise Exception('A channel for first presentation has a t-value over 7! Bad.')

        all_R_x = xr.DataArray(all_R_ttest, dims = ['channels', 'freqs', 'timepoints'], 
                              coords = {'channels': select_channels})

        # *** Second Forgotten - First Forgotten *** #
        all_N_ttest = stats.ttest_rel(session_secondN, session_firstN, axis = 2).statistic

        check_all_N_ttest = abs(all_N_ttest)
        if check_all_N_ttest[check_all_N_ttest>7].shape[0] > 0:
            raise Exception('A channel for first presentation has a t-value over 7! Bad.')

        all_N_x = xr.DataArray(all_N_ttest, dims = ['channels', 'freqs', 'timepoints'], 
                              coords = {'channels': select_channels})
        
        # *** Three Remembered - Three Forgotten *** #
        three_sme = stats.ttest_ind(session_threeR, session_threeN, equal_var = False, axis = 2).statistic

        check_three_sme= abs(three_sme)
        if check_three_sme[check_three_sme>7].shape[0] > 0:
            raise Exception('A channel for first presentation has a t-value over 7! Bad.')

        three_sme_x = xr.DataArray(three_sme, dims = ['channels', 'freqs', 'timepoints'], 
                              coords = {'channels': select_channels})

        if hemisphere == '':
            save_path = '/scratch/radrogue/RepFR1/{}/{}/sme_pow_debug/'.format(region,subject)
        else:
            save_path = '/scratch/radrogue/RepFR1/{}/{}/sme_pow_debug/'.format(hemisphere+'_'+region,subject)

        file_name = '{}_{}_sme_tscore'.format(subj, "first")
        save_numpy(first_x, save_path, file_name)

        file_name = '{}_{}_sme_tscore'.format(subj, "second")
        save_numpy(second_x, save_path, file_name)

        file_name = '{}_{}_sme_tscore'.format(subj, "all_pres")
        save_numpy(all_pres_x, save_path, file_name)

        file_name = '{}_{}_sme_tscore'.format(subj, "all_pres_R")
        save_numpy(all_R_x, save_path, file_name)

        file_name = '{}_{}_sme_tscore'.format(subj, "all_pres_N")
        save_numpy(all_N_x, save_path, file_name)
        
        file_name = '{}_{}_sme_tscore'.format(subj, "three_sme")
        save_numpy(three_sme_x, save_path, file_name)

        print("data saved.")
        print("\n")
        return (subj + ' worked!')
    except Exception as e:
        return (subj + ' failed :( ' + str(e))