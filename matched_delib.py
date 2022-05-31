'''
Created 2/24 by Brandon Katerman * 

new matched deliberation code -- previous titled "bk_md.py"

'''

import os 
from time import time 

import numpy as np 
import pandas as pd 
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import xarray as xr 
import scipy.stats as stats 

from matplotlib import pyplot as plt 
import matplotlib as mpl 
mpl.rcParams['axes.grid'] = False 

import seaborn as sns 
sns.set(color_codes = True)

import cmlreaders 
from cmlreaders import CMLReader, get_data_index 
#from helper_funcs import *
print("imports imported <3 ")


def find_recalls(subj, sess, expmt, n_lists):
    """Return a dataframe with recall start times and all vocalizations from each list.
    
    Classifies correct recalls, repeats, intrusions, and other vocalizations. Also
    finds the differences in timing between consecutive events.
    
    Adapted code from Daniel.
    
    Parameters
    ----------
    subj : str
        Subject code 
    sess : int 
        Session number 
    expmt : str
        experiment code. (ex. ltpRepFR)
    n_lists : int
        number of lists in experiment 
        
        
    Returns
    -------
    out : pandas.DataFrame 
        Dataframe containing all recall and vocalization events in provided session
    """
    
    def flag_intrusions(row,
                    word_list,
                    flag_plis=True):
        """Return whether word is an intrusion (in-list or out-of-list)."""
        if flag_plis:
            curr_words = np.unique(word_list.loc[(word_list.index==row['trial'])].values.ravel()).tolist()
            if np.isin(row['item_name'], curr_words):
                return 0
            else:
                prev_words = np.unique(word_list.loc[(word_list.index<row['trial'])].values.ravel()).tolist()
                return 1 * np.isin(row['item_name'], prev_words)
        else:
            prev_words = np.unique(word_list.loc[(word_list.index<=row['trial'])].values.ravel()).tolist()
            return 0 if np.isin(row['item_name'], prev_words) else 1

    def flag_repeats(words_in):
        """Return whether word has already been said for the current list."""
        words_out = []
        repeated = []
        for word in words_in:
            if word in words_out:
                repeated.append(1)
            else:
                repeated.append(0)
                words_out.append(word)
        return repeated

    def get_prev_rec(row,
                     rec_events,
                     col='study_pos',
                     default=-1):
        """Get value from the previous recall of the current list.

        Returns the default if current word is the first recall,
        or if the previous recall was an out-of-list intrusion.
        """
        if row['rec_pos'] == 1:
            return default

        prev_rec = rec_events.query("(trial=={}) & (rec_pos=={})".format(row['trial'], row['rec_pos']-1)).iloc[0]
        if prev_rec['oli'] == 1:
            return prev_rec[col]
        else:
            return prev_rec[col]

    # Get correct recall and intrusion dataframes for each subject and session.
    
    # save_output = 1
    # overwrite = 0
    # verbose = 1

    # ---------------------
    intrusions = []
    all_recs = []
    subj_sess = '{}_ses{}'.format(subj, sess)
    try:
        # Load data for a session.
        reader = CMLReader(subj, expmt, sess)
        events = reader.load('task_events')
        events.rename(columns={'list': 'trial'}, inplace=True)
        keep_cols = ['subject', 'session', 'trial', 'mstime', 'eegoffset',
                     'item_name', 'serialpos', 'rectime', 'type']
        _keep_cols = [col for col in keep_cols if col in events.columns]
        events = events[_keep_cols]

        # Get study word events during encoding.
        study_events = events.query("(type=='WORD') & (0<trial<{})".format(n_lists)).reset_index(drop=True)
        study_events.drop(columns=['rectime', 'type'], inplace=True)
        word_list = pd.pivot(study_events, index='trial', columns='serialpos', values='item_name')
        all_words = np.unique(word_list.values.ravel()).tolist()

        # Get recall events during retrieval.
        rec_events = events.query("(type==['REC_WORD', 'REC_WORD_VV']) & (0<trial<{})".format(n_lists)).reset_index(drop=True)
        rec_events.drop(columns=['serialpos'], inplace=True)
        rec_events['correct'] = rec_events.apply(lambda x: 1 * np.isin(x['item_name'], word_list.loc[x['trial']]), axis=1)
        rec_events['pli'] = rec_events.apply(lambda x: flag_intrusions(x, word_list, flag_plis=True), axis=1)
        rec_events['oli'] = rec_events.apply(lambda x: flag_intrusions(x, word_list, flag_plis=False), axis=1)
        rec_events['repeat'] = np.concatenate(rec_events.groupby('trial')['item_name'].apply(flag_repeats).tolist())
        rec_events['study_trial'] = -1
        rec_events.loc[rec_events['oli']==0, 'study_trial'] = (rec_events.loc[rec_events['oli']==0, 'item_name']
                                                                         .apply(lambda x: word_list.iloc[np.where(word_list==x)].index[0]))
        rec_events['study_pos'] = -1
        rec_events.loc[rec_events['oli']==0, 'study_pos'] = (rec_events.loc[rec_events['oli']==0, 'item_name']
                                                                       .apply(lambda x: word_list.iloc[np.where(word_list==x)].columns[0]))
        rec_events['rec_pos'] = np.concatenate(rec_events.groupby('trial')['item_name'].apply(lambda x: np.arange(1, len(x)+1).tolist()).tolist())

        # Get some info on the previous recall.
        rec_events['prev_correct'] = rec_events.apply(lambda x: get_prev_rec(x, rec_events, col='correct'), axis=1)
        rec_events['prev_pli'] = rec_events.apply(lambda x: get_prev_rec(x, rec_events, col='pli'), axis=1)
        rec_events['prev_repeat'] = rec_events.apply(lambda x: get_prev_rec(x, rec_events, col='repeat'), axis=1)
        rec_events['prev_study_trial'] = rec_events.apply(lambda x: get_prev_rec(x, rec_events, col='study_trial'), axis=1)
        rec_events['prev_study_pos'] = rec_events.apply(lambda x: get_prev_rec(x, rec_events, col='study_pos'), axis=1)
        rec_events['prev_rectime'] = rec_events.apply(lambda x: get_prev_rec(x, rec_events, col='rectime'), axis=1)

        # Calculate differences between current and previous recalls.
        rec_events['study_trial_diff'] = -1
        rec_events.loc[(rec_events['prev_correct']!=-1), 'study_trial_diff'] = (rec_events.loc[(rec_events['prev_correct']!=-1), 'study_trial'] -
                                                                                rec_events.loc[(rec_events['prev_correct']!=-1), 'prev_study_trial'])
        rec_events['study_pos_diff'] = -1
        rec_events.loc[(rec_events['prev_correct']!=-1), 'study_pos_diff'] = (rec_events.loc[(rec_events['prev_correct']!=-1), 'study_pos'] -
                                                                              rec_events.loc[(rec_events['prev_correct']!=-1), 'prev_study_pos'])
        rec_events['rectime_diff'] = -1
        rec_events.loc[(rec_events['prev_correct']!=-1), 'rectime_diff'] = (rec_events.loc[(rec_events['prev_correct']!=-1), 'rectime'] -
                                                                            rec_events.loc[(rec_events['prev_correct']!=-1), 'prev_rectime'])
#         # Get vocalziations during retrieval ##BK changes
#         voc_events = events.query("type =='REC_WORD_VV' & (0<trial<{})".format(n_lists)).reset_index(drop=True)
#         if voc_events.shape[0] > 0:
#             voc_events.drop(columns=['serialpos'], inplace = True)
#             voc_events['correct'] = 0
#             voc_events['pli'] = 0
#             voc_events['oli'] = 0
#             voc_events['repeat'] = 0
#             voc_events['study_trial'] = -1
#             #voc_events.loc[voc_events['oli']==0, 'study_trial'] = (voc_events.loc[voc_events['oli']==0, 'item_name'].apply(lambda x: word_list.iloc[np.where(word_list==x)].index[0]))
#             voc_events['study_pos'] = -1
#             #voc_events.loc[voc_events['oli']==0, 'study_pos'] = (voc_events.loc[voc_events['oli']==0, 'item_name'].apply(lambda x: word_list.iloc[np.where(word_list==x)].columns[0]))
#             voc_events['rec_pos'] = np.concatenate(voc_events.groupby('trial')['item_name'].apply(lambda x: np.arange(1, len(x)+1).tolist()).tolist())

#             # Get some info on the previous recall.
#             voc_events['prev_correct'] = voc_events.apply(lambda x: get_prev_rec(x, voc_events, col='correct'), axis=1)
#             voc_events['prev_pli'] = voc_events.apply(lambda x: get_prev_rec(x, voc_events, col='pli'), axis=1)
#             voc_events['prev_repeat'] = voc_events.apply(lambda x: get_prev_rec(x, voc_events, col='repeat'), axis=1)
#             voc_events['prev_study_trial'] = voc_events.apply(lambda x: get_prev_rec(x, voc_events, col='study_trial'), axis=1)
#             voc_events['prev_study_pos'] = voc_events.apply(lambda x: get_prev_rec(x, voc_events, col='study_pos'), axis=1)
#             voc_events['prev_rectime'] = voc_events.apply(lambda x: get_prev_rec(x, voc_events, col='rectime'), axis=1)
#         elif voc_events.shape[0] == 0:
#             voc_events = events.query("(type == 'None')")

#     #             # Get non-repeat, correct recalls.
#     #             corr_recs.append(rec_events.query("(repeat==0) & (correct==1)").reset_index(drop=True))

        # concat all recall + vocalziation events 
        all_recs.append(pd.concat([rec_events.reset_index(drop=True),
#                                    rec_events.query("(correct==1)").reset_index(drop=True),
#                                    voc_events,
#                                    rec_events.query("(pli==1)").reset_index(drop=True),
#                                    rec_events.query("(oli==1)").reset_index(drop=True),
                                   events.query("(type=='REC_START') & (0<trial<{})".format(n_lists))],
                                   axis=0).sort_values('eegoffset').reset_index(drop=True))

#         # Get non-repeat, in-list intrusions.
#         intrusions.append(rec_events.query("(repeat==0) & (pli==1)").reset_index(drop=True))
    except Exception as e:
        print(e)
        #failed_sessions.append(subj_sess)
    
#     if not 'subj_sess' in rec_events.columns:
#         rec_events.insert(0, 'subj_sess', rec_events.apply(lambda x: '{}_ses{}'.format(x['subject'], x['session']), axis=1))

    all_recs = pd.concat(all_recs).reset_index(drop=True)
     #plis = pd.concat(intrusions).reset_index(drop=True)

    all_recs.insert(0, 'subj_sess', all_recs.apply(lambda x: '{}_ses{}'.format(x['subject'], x['session']), axis=1))
    #plis.insert(0, 'subj_sess', plis.apply(lambda x: '{}_ses{}'.format(x['subject'], x['session']), axis=1))

    # Rename columns, as needed.
    col_map = {'trial': 'list'}
    all_recs.rename(columns=col_map, inplace=True)

    # # Save files.
    # filename = op.join(proj_dir, 'analysis', 'events', '{}-plis.pkl'.format(expmt))
    # if save_output:
    #     filename = op.join(proj_dir, 'analysis', 'events', '{}-corr_recs.pkl'.format(expmt))
    #     if overwrite or not op.exists(filename):
    #         dio.save_pickle(corr_recs, filename, verbose)
    #     filename = op.join(proj_dir, 'analysis', 'events', '{}-plis.pkl'.format(expmt))
    #     if overwrite or not op.exists(filename):
    #         dio.save_pickle(plis, filename, verbose)

    #print('all_recs: {}'.format(all_recs.shape))
    
    return all_recs

def find_silence(recall_events,
                 silence_len=1400,
                 post_rec_distance=1400,
                 pre_rec_distance=2833,
                 sr=2048):
    '''
    Find all periods of silence in a session given a dataframe of all recall and vocalization events.
    
    Parameters
    ----------
    recall_events: dataframe
        Includes all recall + vocalization events in a session.
    silence_len: int 
        Silence window duration, in ms.
        
    post_rec_distance: int 
        duration after a recall to exclude from marked silence 
        
    pre_rec_distance: int 
        duration before a recall to exclude from marked silence 

    sr : int
        Sampling rate of the EEG.
                
    Returns
    -------
    silence_df: dataframe
        Contains all qualifying periods of silence in the session.
        Each silence window is indicated by SILENCE_START and SILENCE_STOP.
    '''
    #inital dataframes for the start and stop times of periods of silence 
    start_df = pd.DataFrame([], columns = ['subject', 'session', 'list', 'eegoffset' ,'mstime', 'rectime'])
    stop_df = pd.DataFrame([], columns = ['subject', 'session', 'list', 'eegoffset' ,'mstime', 'rectime'])


    #get information for each recall event one at a time (remove REC_START events because not real recalls)
    #first recall of any recall window should not be included
    for x in range(recall_events.query("type != 'REC_START'").shape[0]-1):

        recall_events = recall_events.query("type != 'REC_START'") 
        
        #current rec information
        silence_subj = recall_events.iloc[x].subject
        silence_sess = recall_events.iloc[x].session
        silence_l = recall_events.iloc[x].list


        #exclude periods of silence that are followed by a vocalization because it could indicate that participants spoke the current word for more than one second 

        #mark starting period of silence 1400 ms after voc. onset to be positive we are not contaminating with speech
        silence_rec_start = recall_events.iloc[x].rectime + post_rec_distance
        silence_ms_start = recall_events.iloc[x].mstime + (post_rec_distance) 
        silence_eegoffset = recall_events.iloc[x].eegoffset + (post_rec_distance * (sr / 1000.))
        #really big time prior to a recall so that we don't get duplicate recall + deliberation periods (overlapping events by milliseconds)
        silence_rec_stop = recall_events.iloc[x+1].rectime - pre_rec_distance
        silence_ms_stop = recall_events.iloc[x+1].mstime - (pre_rec_distance)
        silence_eegoffset_stop = recall_events.iloc[x+1].eegoffset - (pre_rec_distance * (sr / 1000.)) 

        #real period of silence -- the start time doesn't overlap into a stop time -- this could happen if participant is recalling in a quick succession 
        if silence_rec_start < silence_rec_stop:

            silence_rectime_len = (silence_rec_stop - silence_rec_start)


            #if the length of the silence period is greater than the length x 2, we should split it into two periods of silence 
            if silence_rectime_len >= (silence_len*2):

                #need to make sure that mstime, rectime, and eegoffset has the same amount of splits or it'll fuck everything up
                rec_time_range = np.arange(silence_rec_start, silence_rec_stop+1)
                ms_time_range = np.arange(silence_ms_start, silence_ms_stop+1)
                eegoffset_time_range = np.arange(silence_eegoffset, silence_eegoffset_stop+1)

                num_splits = int(silence_rectime_len/silence_len)

                #split them by the same number (this should work I think)
                rec_splits = np.array_split(rec_time_range, num_splits)
                ms_splits = np.array_split(ms_time_range, num_splits)
                eeg_splits = np.array_split(eegoffset_time_range, num_splits)
            
                for n in range(num_splits):
                    #loop through the range and save all relevant values 
                    silence_rec_start = min(rec_splits[n])
                    silence_rec_stop = max(rec_splits[n])

                    silence_ms_start = min(ms_splits[n])
                    silence_ms_stop = max(ms_splits[n])

                    silence_eegoffset = min(eeg_splits[n]) ##check if these values are correct 
                    silence_eegoffset_stop = max(eeg_splits[n])

                    start_event = 'SILENCE_START'
                    stop_event = 'SILENCE_STOP'

                    #update eegoffset + mstime values to be true to the split rectimes
                    start_event_df = pd.DataFrame([[silence_subj, silence_sess, silence_l ,int(silence_eegoffset), silence_ms_start, int(silence_rec_start), start_event]], columns =['subject', 'session', 'list', 'eegoffset' ,'mstime', 'rectime', 'type']) 
                    start_df = start_df.append(start_event_df)

                    stop_event_df = pd.DataFrame([[silence_subj, silence_sess, silence_l ,int(silence_eegoffset_stop), silence_ms_stop, int(silence_rec_stop), stop_event]], columns =['subject', 'session', 'list', 'eegoffset' ,'mstime', 'rectime', 'type']) 
                    stop_df = stop_df.append(stop_event_df)


            #get any period of silence that is less than the desired silence length x 2 
            elif (silence_len*2) > silence_rectime_len >= silence_len :



                start_event = 'SILENCE_START'
                stop_event = 'SILENCE_STOP'


                start_event_df = pd.DataFrame([[silence_subj, silence_sess, silence_l ,int(silence_eegoffset), silence_ms_start, int(silence_rec_start), start_event]], columns =['subject', 'session', 'list', 'eegoffset' ,'mstime', 'rectime', 'type']) 
                start_df = start_df.append(start_event_df)

                stop_event_df = pd.DataFrame([[silence_subj, silence_sess, silence_l ,int(silence_eegoffset_stop), silence_ms_stop, int(silence_rec_stop), stop_event]], columns =['subject', 'session', 'list', 'eegoffset' ,'mstime', 'rectime', 'type']) 
                stop_df = stop_df.append(stop_event_df)


        

        else:
            pass 


    
    rec_starts = recall_events.query("type == 'REC_START'")
    remaining_cols = ['subject', 'session', 'list', 'eegoffset' ,'item_name','mstime', 'rectime', 'type'] #turn this into a real variable to input into above dataframes
    rec_starts = rec_starts[remaining_cols]

    #concat together all the silence start, stop, and rec start dfs + reset the indexes 
    silence_df = pd.concat([start_df, stop_df, rec_starts]).sort_values('eegoffset')  
    silence_df.reset_index(inplace = True, drop = True)  
    
    return silence_df

def find_time_matches(recall_events,
                      silence_df,
                      n_lists,
                      list_thres=25,
                      proximity_buffer=5000):
    '''Find matched silences for each correct recall event.
    
    Parameters 
    ----------
    recall_events : dataframe
        Dataframe of correct recall events for which you want
        to find matched silences.
    silence_df : dataframe
        Dataframe of all qualifying silence intervals.
    n_lists : int
        Number of lists in the experiment.
    proximity_buffer : int 
        Time in ms around exact rectime to still match silence.
            
    Returns
    -------
    sorted_df : dataframe 
        DataFrame containing all recall events with a new columns including 
        "match_index" that contains a list of all indexes in relation to the 
        "silence_df" DataFrame that qualified as matched silence. 
        Matched silence is qualifying if the rectime of a given recall event 
        falls within the SILENCE_START and SILENCE_STOP time range or if it's 
        within 2 seconds of the SILENCE_START and SILENCE_STOP times. 
    '''
    #make it so it looks equal parts back and forward first 
    #use np.random if it's below .5 look back if it's above look forward 
    def find_adjacent_lists(current_list,
                            n_lists):
        '''
        create a list of the order in which we should check lists (nearest first) for qualifying periods of silence 

        Parameters
        ----------
        current_list : int
            The list that the present recall you want to match is from.
        n_lists : int
            The number of lists in a session.
        list_thres : int
            List number that qualifying periods of silence should be dropped.
        '''
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
        
        n_lists = np.arange(1,n_lists)
        none_target_lists = [n for n in n_lists if n != current_list]

        if np.random.rand() > 0.5:
            none_target_lists = np.flip(none_target_lists)
        else:
            pass 
        
        list_search_order = []

        for x in range(len(none_target_lists)):
            nearest_list = find_nearest(none_target_lists, current_list)
            list_search_order.append(nearest_list)
            none_target_lists = [n for n in none_target_lists if n not in list_search_order]

        list_search_order = np.asarray(list_search_order)
        total_possible_lists = n_lists.shape[0] - 1
        total_found_list = list_search_order.shape[0]
        if total_possible_lists != total_found_list:
            print("I think there's a missing list somewhere :( ")
            print("total number of possible lists: "+str(total_possible_lists))
            print("total number of actual lists added to array: "+str(total_found_list))
        
        return list_search_order

    #should already be filtered 
    #recall_events = recall_events.query("type != 'REC_START'")
    #recall_events = recall_events.query("type != 'REC_WORD_VV'")
    
    #dictionary to keep track of all the indexes of the periods of silence 
    match_dict = {}
    for x in range(recall_events.shape[0]):
        match_dict['{0}'.format(str(x))] = []

    big_match_list = []

    # Count the number of eligible recall events for each recall.
    matched_events_counter = np.zeros(recall_events.shape[0])

    for x in range(recall_events.shape[0]):

        choice_event = recall_events.iloc[x] #event that we want to find a matched period of silence for
        choice_event_list = choice_event.list
        choice_rectime = choice_event.rectime

        # Look match rectimes that are within this Xms range.
        choice_list = choice_event.list
        list_pick_order = find_adjacent_lists(choice_list, n_lists) 
        
        list_pick_order = [l for l in list_pick_order if abs(l-choice_list) < list_thres]

        match_list = []

        for l in list_pick_order:
            list_df = silence_df[silence_df['list'] == l]

            #look for exact matches first 
            for START, STOP in zip(list_df.query("type == 'SILENCE_START'").rectime,
                                   list_df.query("type == 'SILENCE_STOP'").rectime):
                match = False 
                #print(START,STOP)
                
                if choice_rectime == START:
                    print("exact match")
                    #print('match for recall index '+str(x))

                #adjust buffer implementation 
                
                    match = True
                    
                    if match == True:
                        
                        match_list.append(int(list_df[list_df['rectime'] == START].index.values))
                    
                   
                    matched_events_counter[x] += 1 
                    
                    
                if START in range(choice_rectime, choice_rectime + proximity_buffer):
                    #print("within 2 seconds (+)")

                    match = True
            #look for matches within buffer range 
            for START, STOP in zip(list_df.query("type == 'SILENCE_START'").rectime,
                                   list_df.query("type == 'SILENCE_STOP'").rectime):
                
                match = False 
                
                if choice_rectime < choice_rectime - proximity_buffer:
                    if START in range(choice_rectime, choice_rectime - proximity_buffer):
                        #print("within 2 seconds (-)")

                        match = True
                        
                if choice_rectime > choice_rectime - proximity_buffer:
                     if START in range(choice_rectime - proximity_buffer ,choice_rectime):
                        #print("within 2 seconds (-)")

                        match = True


                if match == True:
                    match_list.append(int(list_df[list_df['rectime'] == START].index.values))
                    
                   
                    matched_events_counter[x] += 1 
                else:
                    pass 


                #silence_df.loc[silence_df['rectime'] == START]


                #print(START, STOP)

        #if i save indexes as a dict 
        match_dict[str(x)] = match_list


        #save it in a master list 
        big_match_list.append(match_list)
        #matched_events_counter[x] = len(match_list)


    #if i had the information as a separte column in the recall_events df 
    recall_events['match_index'] = big_match_list
    num_matches = [int(len(l)) for l in big_match_list]
    recall_events['num_match'] = list(matched_events_counter)
    
    

    #only hold onto events that have matches
    matchable_events = pd.DataFrame([])
    for x in range(recall_events.shape[0]):
        present_event = recall_events.iloc[x]


        if present_event.num_match > 0:
            matchable_events = matchable_events.append(present_event)
            

    sorted_df = matchable_events.sort_values(by = 'num_match', ascending = True)
    sorted_df['matched'] = np.zeros(sorted_df.shape[0], dtype = int)
    sorted_df.reset_index(inplace=True, drop=True)

    return sorted_df

def Create_Matched_Deliberations(df, silence_df, print_matching_index = False):
    


    '''
    Create dataframe of desired recall events and paired matched deliberation period. 
    As each recall event is paired with a period of silence, that recall event is marked as matched and that specific period of silence is removed from all remaining unmatched recalls "match_index" column.


    Parameters 
    ----------
    df: pandas.DataFrame 
    recall dataframe that includes "match_index" of qualifying matched silence periods in relation to silence_df created from "find_time_matches" function

     silence_df: pandas.DataFrame
                Dataframe created from find_silence function -- all qualifying periods of silence in a session 


    Returns
    -------
    matched_delibs: pandas.DataFrame 
                    Dataframe containing all recall and paired matched periods of silence 

    '''



    event_errors = []

    matched_delibs = pd.DataFrame([], columns = list(silence_df.columns))
    rec_df_len = df.shape[0]

    for x in range(rec_df_len):
        try:

            df = df[df['num_match'] != 0]
            if print_matching_index == True:
                print("matching event index "+str(x))
            min_key = df.iloc[0].match_index[0]
            current_event = df.iloc[0]

            #changed from sorted_rec_df in other code -- see why that works but this doesnt 
            matched_count = np.zeros(df.shape[0])
            matched_count[0] += 1 


            matched_delibs = matched_delibs.append(current_event[silence_df.columns])
            matched_delibs = matched_delibs.append(silence_df.loc[min_key])
            #silence stops 
            #matched_delibs = matched_delibs.append(silence_df.loc[min_key+1])



            #update and remove matches 

            match_index = min_key

            new_match_events = pd.DataFrame([])

            #remove the current matched silence event from all match_index fields and update the Len Matches for each recall event 
            match_holder = []
            match_len_holder = []

            event_len = df.shape[0]

            for x in range(event_len):
                present_event = df.iloc[x]
                matches = present_event.match_index

                new_matches = [m for m in matches if m != match_index]
                new_match_len = len(new_matches)

                match_holder.append(new_matches)
                match_len_holder.append(new_match_len)

            #update 
            df['match_index'] = match_holder
            df['num_match'] = match_len_holder
            df['matched'] = matched_count 


            df = df[df['num_match'] != 0]
            df = df[df['matched'] != 1]
            df = df.sort_values(by = 'num_match', ascending = True)
                #df.reset_index(inplace = True, drop = True)


                #sorted_rec_df = update_and_remove_matches(sorted_rec_df, min_key)

                #rec_df_len = sorted_rec_df.shape[0]

        except Exception as e:
            event_errors.append(e)
            #print("error with event index "+str(x))
            #print(e)

    matched_delibs.reset_index(inplace = True, drop = True)


    return matched_delibs 


def matched_delibs(subj, 
                   sess, 
                   exp, 
                   n_lists,
                   log_freqs,
                   wave_number,
                   rec_window,
                   sr,
                  pre_voc_exclusion = 1300,
                   voca_dur = 500):
    
    
    
    all_recs = find_recalls(subj, sess, exp, n_lists)
    
    rec_len = np.diff(rec_window)[0]
    #remove wave buffer for wavelets per Mike's request 3/10
    half_wav = ((1000 / log_freqs.min()) * wave_number) * 0.5
       
    #post_rec_distance = voca_dur 
    #pre_rec_distance = pre_voc_exclusion

    post_rec_distance = voca_dur + half_wav
    pre_rec_distance = pre_voc_exclusion + half_wav
    
    silence = find_silence(all_recs, 
                          silence_len = rec_len,
                          post_rec_distance = post_rec_distance,
                          pre_rec_distance = pre_rec_distance,
                          sr = sr)
    
    print('Total vocalizations found: '+str(all_recs.shape[0]))
    #print("Total silences found: "+str(silence.shape[0]))
    
    
    #inter_rec_min = abs(rec_window[0]) + voca_dur + half_wav
    inter_rec_min = abs(rec_window[0]) + voca_dur
    qry1 = ("(rectime_diff>={})".format(inter_rec_min))
    choice_recall = all_recs.query(qry1)
    
    qry2 = "(correct==1) & (repeat==0)"
    choice_recall = choice_recall.query(qry2)
    
    total_recalls = choice_recall.shape[0]

    print("total filtered recs: "+str(total_recalls))
    print("Total silences found: "+str(silence.shape[0]))

    
    
    time_matches = find_time_matches(choice_recall, silence, n_lists, list_thres = 20, proximity_buffer = 5000)
    md = Create_Matched_Deliberations(time_matches, silence)
    print('# matched periods of silence: {}'.format(md.shape[0] / 2))
    
    md_d = {"all_recs":all_recs,
         'silence': silence, 
         'choice_recall': choice_recall,
         'matched_events': time_matches,
         'delibs':md}
    
    return md_d
        
    
    
print("functions imported")
    
    