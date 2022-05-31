import os
import pickle
import cmlreaders as cml
import pandas as pd
import xarray as xr
from joblib import dump, load
import scipy as sp
import numpy as np
import glob
import sys
from sklearn.metrics import roc_auc_score
from sklearn import clone
from sklearn.model_selection import LeaveOneGroupOut
from ptsa.data.timeseries import TimeSeries

def process_events(events_df):
    pd.options.mode.chained_assignment = None  # default='warn'
    events_df = events_df[events_df['list'] > 0]
    # processing events
    events_df['subject'] = events_df['subject'].astype(str)
    if 'item' not in events_df.columns:
        events_df.rename(columns={'item_name': 'item'}, inplace=True)
    if 'category_num' not in events_df.columns:
        events_df.rename(columns={'categoryNum': 'category_num'}, inplace=True)
    events_df['item'] = events_df.item.str.strip().str.lower()
    events_df.loc[events_df['item'] == 'axe', 'item'] = 'ax'
    return events_df

def process_events_retrieval(events):
    # preprocessing of retrieval events from DJH code
    events = events[events['list'] >= 0]
    if 'item' not in events.columns:
        events.rename(columns={'item_name': 'item'}, inplace=True)   
    events['item'] = events.item.str.strip().str.lower()
    events.loc[events['item'] == 'axe', 'item'] = 'ax'

    events['pirt'] = events.groupby(['session', 'list'])['rectime'].diff().fillna(events['rectime'])
    events['repeat'] = events.duplicated(subset=['session', 'list', 'item'])
    use_idx = (events['pirt'] > 1500) & (events['repeat'] == 0)# & (events['intrusion'] == 0)  
    return events

def process_events_countdown(events):
    events = events[events['list'] >= 0]

    events['pirt'] = events.groupby(['session', 'list'])['rectime'].diff().fillna(events['rectime'])
    events['repeat'] = events.duplicated(subset=['session', 'list'])
    use_idx = (events['pirt'] > 1500) #& (events['repeat'] == 0) & (events['intrusion'] == 0)  
    return events

def load_subj_data_repFR(df_subj, period='encoding', 
                   preproc_data_dir='preproc_data/catFR1/',
                  rename=False, as_array=False, identifier=0,
                  presentation = 0, repeat = 0):
    # presentation = 0 means all presentations. 1 means 1st pres only, 2 mean 2nd, and 3 means 3rd
    # repeat = 0 means all encoding. 1 means 1p only, 2 means 2p, 3 means 3p.
    z_pow_stacked_list = []
                 
    for _, df_sess in df_subj.iterrows():
        preproc_fp = preproc_data_dir + period + '/' + df_sess['subject'] + '_' + str(df_sess['session']) + '.h5'
        if os.path.exists(preproc_fp):
            try:
                z_pow_stacked = TimeSeries.from_hdf(preproc_fp)
                z_pow_stacked_list.append(z_pow_stacked)
            except:
                print('error', df_sess['session'], sys.exc_info())
                continue
    if len(z_pow_stacked_list) == 0:
        return False, False, False

    z_pow_stacked = xr.concat(z_pow_stacked_list, dim='event')
    
    if period == 'encoding':
    
        # need to get presentation number (e.g. 3rd time 3p shown) for RepFR1

        # session_events is df that contains channel_num and item_num cols, make that here
        session_events = pd.DataFrame(data={'item_num':z_pow_stacked.item_num.values,
                                            'channel_num':np.zeros(len(z_pow_stacked.item_num.values))}) 
        # channel_num is only required for dependencies in SWR code so just add dummy column here  satisfy function below)
        list_num_key = z_pow_stacked.list.values
        session_name_array = z_pow_stacked.session.values

        import sys; sys.path.append('/home1/john/johnModules')
        from SWRmodule import getRepFRPresentationArray
        presentation_array = getRepFRPresentationArray(session_name_array,list_num_key,session_events) 
        z_pow_stacked = z_pow_stacked.assign_coords(presentations=('event', presentation_array)) # add this to event

        # now select based on presentation and repeat inputs
        if ((presentation == 0) & (repeat == 0)):
            z_pow_stacked = z_pow_stacked            
        elif presentation == 0: # select by repeat only 
            z_pow_stacked = z_pow_stacked[z_pow_stacked.repeats == repeat]
        elif repeat == 0: # select by repeat only 
            z_pow_stacked = z_pow_stacked[z_pow_stacked.presentations == presentation]
        else:
            z_pow_stacked = z_pow_stacked[((z_pow_stacked.presentations == presentation) & \
                                           (z_pow_stacked.repeats == repeat))]    
    
    #has to come before rename because indexes doesn't change coordinate names for some reason
    if as_array:
        z_pow_stacked = xr.DataArray(z_pow_stacked, coords=z_pow_stacked.coords)
    
    if rename:
        z_pow_stacked.indexes['event'].names = [name+identifier for name in z_pow_stacked.indexes['event'].names]
        z_pow_stacked = z_pow_stacked.rename({'event': 'event_'+identifier})
        
    return z_pow_stacked

def load_subj_distractor_data(df_subj, preproc_data_dir='preproc_data/catFR1/'):
    z_pow_stacked_list = []
    distractor_events_df_list = []
    word_events_df_list = []
    rec_events_df_list = []
    for _, df_sess in df_subj.iterrows():
        distractor_preproc_fp = preproc_data_dir + 'distractor/' + df_sess['subject'] + '_' + str(df_sess['session']) + '.pkl'
        encoding_preproc_fp = preproc_data_dir + 'encoding/' + df_sess['subject'] + '_' + str(df_sess['session']) + '.pkl'
        if os.path.exists(distractor_preproc_fp) and os.path.exists(encoding_preproc_fp):
            try:
                z_pow_stacked, distractor_events_df = pickle.load(open(distractor_preproc_fp, 'rb'))
                z_pow_stacked = z_pow_stacked.assign_coords(
                    session=("event", distractor_events_df.session),
                    session_list=("event", 
                                  distractor_events_df.session.astype(str) + '_' + 
                                  distractor_events_df.list.astype(str))
                )
                z_pow_stacked_list.append(z_pow_stacked)
                distractor_events_df_list.append(distractor_events_df)
                _, word_events = pickle.load(open(encoding_preproc_fp, 'rb'))
                word_events_df_list.append(word_events)
                reader = cml.CMLReader(subject=df_sess['subject'], 
                           experiment=df_sess['experiment'], session=df_sess['session'],
                           localization=df_sess['localization'], montage=df_sess['montage'])
                rec_events = reader.load('task_events').query('type == ["REC_WORD"]')
                rec_events = process_events(rec_events)
                rec_events_df_list.append(rec_events)
            except:
                print('error', df_sess['session'], sys.exc_info())
                continue
    if len(z_pow_stacked_list) == 0:
        return False, False, False
    distractor_events_df = pd.concat(distractor_events_df_list)
    z_pow_stacked = xr.concat(z_pow_stacked_list, dim='event')
    word_events_df = pd.concat(word_events_df_list)
    word_events_df['session_list'] = word_events_df.session.astype(str) + '_' + word_events_df.list.astype(str)
    rec_events_df = pd.concat(rec_events_df_list)
    
    return z_pow_stacked, word_events_df, rec_events_df, distractor_events_df

def load_subj_test_period_data(df_subj, period='distractor', preproc_data_dir='preproc_data/catFR1/'):
    z_pow_stacked_list = []
    test_period_events_df_list = []
    word_events_df_list = []
    rec_events_df_list = []
    for _, df_sess in df_subj.iterrows():
        test_period_preproc_fp = preproc_data_dir + period + '/' + df_sess['subject'] + '_' + str(df_sess['session']) + '.pkl'
        encoding_preproc_fp = preproc_data_dir + 'encoding/' + df_sess['subject'] + '_' + str(df_sess['session']) + '.pkl'
        if os.path.exists(test_period_preproc_fp) and os.path.exists(encoding_preproc_fp):
            try:
                z_pow_stacked, test_period_events_df = pickle.load(open(test_period_preproc_fp, 'rb'))
                z_pow_stacked = z_pow_stacked.assign_coords(
                    session=("event", test_period_events_df.session),
                    session_list=("event", 
                                  test_period_events_df.session.astype(str) + '_' + 
                                  test_period_events_df.list.astype(str))
                )
                z_pow_stacked_list.append(z_pow_stacked)
                test_period_events_df_list.append(test_period_events_df)
                _, word_events = pickle.load(open(encoding_preproc_fp, 'rb'))
                word_events_df_list.append(word_events)
                reader = cml.CMLReader(subject=df_sess['subject'], 
                           experiment=df_sess['experiment'], session=df_sess['session'],
                           localization=df_sess['localization'], montage=df_sess['montage'])
                rec_events = reader.load('task_events').query('type == ["REC_WORD"]')
                rec_events = process_events(rec_events)
                rec_events_df_list.append(rec_events)
            except:
                print('error', df_sess['session'], sys.exc_info())
                continue
    if len(z_pow_stacked_list) == 0:
        return False, False, False, False
    test_period_events_df = pd.concat(test_period_events_df_list)
    z_pow_stacked = xr.concat(z_pow_stacked_list, dim='event')
    word_events_df = pd.concat(word_events_df_list)
    word_events_df['session_list'] = word_events_df.session.astype(str) + '_' + word_events_df.list.astype(str)
    rec_events_df = pd.concat(rec_events_df_list)
    
    return z_pow_stacked, word_events_df, rec_events_df, test_period_events_df

def load_subj_rec_data(df_subj, preproc_data_dir='preproc_data/catFR1/'):
    z_pow_stacked_list = []
    word_events_df_list = []
    rec_events_df_list = []
    for _, df_sess in df_subj.iterrows():
        rec_preproc_fp = preproc_data_dir + 'retrieval/' + df_sess['subject'] + '_' + str(df_sess['session']) + '.pkl'
        encoding_preproc_fp = preproc_data_dir + 'encoding/' + df_sess['subject'] + '_' + str(df_sess['session']) + '.pkl'
        if os.path.exists(rec_preproc_fp) and os.path.exists(encoding_preproc_fp):
            try:
                z_pow_stacked, rec_events_df = pickle.load(open(rec_preproc_fp, 'rb'))
                z_pow_stacked = z_pow_stacked.assign_coords(
                    session=("event", rec_events_df.session),
                    session_list=("event", 
                                  rec_events_df.session.astype(str) + '_' + 
                                  rec_events_df.list.astype(str))
                )
                z_pow_stacked_list.append(z_pow_stacked)
                rec_events_df_list.append(rec_events_df)
                _, word_events = pickle.load(open(encoding_preproc_fp, 'rb'))
                word_events_df_list.append(word_events)
            except:
                print('error', df_sess['session'], sys.exc_info())
                continue
    if len(z_pow_stacked_list) == 0:
        return False, False, False
    rec_events_df = pd.concat(rec_events_df_list)
    z_pow_stacked = xr.concat(z_pow_stacked_list, dim='event')
    word_events_df = pd.concat(word_events_df_list)
    word_events_df['session_list'] = word_events_df.session.astype(str) + '_' + word_events_df.list.astype(str)
    rec_events_df = pd.concat(rec_events_df_list)
    
    return z_pow_stacked, word_events_df, rec_events_df

def train_models_cv(df_subj, clf, word2vec_cat_df, cv, vec_col_type='item', 
                    clf_dir='clfs_lolo/', preproc_dir='preproc_data/catFR1/encoding/', force=True):
    subject = df_subj.subject.iloc[0]
    clf_fp = clf_dir + subject + '_' + vec_col_type + '_clf_1.joblib'
    if os.path.exists(clf_fp) and not force:
        return True
    
    if vec_col_type == 'item':
        use_vec_cols = ['vec_' + str(i) for i in range(300)]
    else:
        use_vec_cols = ['mean_vec_' + str(i) for i in range(300)]
    
    print(clf_dir)
    z_pow_stacked_list = []
    events_df_list = []
    
    for _, df_sess in df_subj.iterrows():
        preproc_fp = preproc_dir + df_sess['subject'] + '_' + str(df_sess['session']) + '.pkl'
        if os.path.exists(preproc_fp):
            try:
                z_pow_stacked, events_df = pickle.load(open(preproc_fp, 'rb'))
                z_pow_stacked = z_pow_stacked.assign_coords(session=("event", events_df.session))
                z_pow_stacked_list.append(z_pow_stacked)
                events_df_list.append(events_df)
            except:
                print('error', df_sess['session'], sys.exc_info())
                continue
    if not len(z_pow_stacked_list) == 0:
        events_df = pd.concat(events_df_list)
        merge_on = ['item', 'category', 'category_num']
        if df_sess['experiment'] == "FR1":
            merge_on = 'item'
        events_df = events_df.merge(word2vec_cat_df, on=merge_on, how='left')
        z_pow_stacked = xr.concat(z_pow_stacked_list, dim='event')
        session_list = events_df['session'].astype(str) + '_' + events_df['list'].astype(str)

        fold_ind = 1
        for train_index, test_index in cv.split(events_df[use_vec_cols], groups=session_list):
            print(fold_ind)

            train_vecs = events_df.iloc[train_index].loc[:, use_vec_cols]
            train_pow_all = z_pow_stacked[train_index, :]
            print(train_vecs.shape, train_pow_all.shape)

            clf.fit(train_vecs, train_pow_all)
            clf_fp = clf_dir + subject + '_' + vec_col_type + '_clf_' + str(fold_ind) + '.joblib'
            dump((clf, (train_index, test_index)), clf_fp)
            fold_ind += 1

def train_generalized_models_cv(df_subj, estimator, word2vec_cat_df, vec_col_type='item', clf_dir='clfs_lolo/', preproc_dir='preproc_data/catFR1/encoding/', force=True, force_fold=True):
    subject = df_subj.subject.iloc[0]
    cv = LeaveOneGroupOut()
    if 'model_selection' in str(type(estimator)):
        estimator_class = estimator.estimator.__class__
    elif 'pipeline' in str(type(estimator)):
        pipeline = True
        estimator_class = estimator['estimator'].__class__
    else:
        pipeline = False
        estimator_class = estimator.__class__
    if 'himalaya' in str(estimator_class):
        himalaya = True
        print(estimator.__class__)
    else:
        himalaya = False
    estimator_name = estimator_class.__name__
    print(estimator_name)
    clf_fp = clf_dir + estimator_name + '_' + subject + '_' + vec_col_type + '_clf_1.joblib'
    if os.path.exists(clf_fp) and not force:
        return True
    
    if vec_col_type == 'item':
        use_vec_cols = ['vec_' + str(i) for i in range(300)]
    else:
        use_vec_cols = ['mean_vec_' + str(i) for i in range(300)]
    
    print(clf_dir)
    z_pow_stacked_list = []
    events_df_list = []
    
    for _, df_sess in df_subj.iterrows():
        preproc_fp = preproc_dir + df_sess['subject'] + '_' + str(df_sess['session']) + '.pkl'
        if os.path.exists(preproc_fp):
            try:
                z_pow_stacked, events_df = pickle.load(open(preproc_fp, 'rb'))
                z_pow_stacked = z_pow_stacked.assign_coords(session=("event", events_df.session))
                z_pow_stacked_list.append(z_pow_stacked)
                events_df_list.append(events_df)
            except:
                print('error', df_sess['session'], sys.exc_info())
                continue
    if not len(z_pow_stacked_list) == 0:
        events_df = pd.concat(events_df_list)
        merge_on = ['item', 'category', 'category_num']
        if df_sess['experiment'] == "FR1":
            merge_on = 'item'
        events_df = events_df.merge(word2vec_cat_df, on=merge_on, how='left')
        z_pow_stacked = xr.concat(z_pow_stacked_list, dim='event')
        session_list = events_df['session'].astype(str) + '_' + events_df['list'].astype(str)

        fold_ind = 1
        for train_index, test_index in cv.split(X=events_df[use_vec_cols], groups=session_list):
            clfs = {}
            clf_fp = clf_dir + estimator_name + '_' + subject + '_' + vec_col_type + '_clf_' + str(fold_ind) + '.joblib'
            print(clf_fp)
            if os.path.exists(clf_fp) and not force_fold:
                print('found ' + clf_fp)
                fold_ind += 1
                continue
            print(fold_ind)

            train_vecs = events_df.iloc[train_index].loc[:, use_vec_cols]
            train_pow_all = z_pow_stacked[train_index, :]
            print(train_vecs.shape, train_pow_all.shape)
    
            print(estimator)
            if himalaya:
                search = clone(estimator)
                if pipeline:
                    search.set_params(estimator__cv=cv.split(X=train_vecs, groups=session_list[train_index]))
                else:
                    search.set_params(cv=cv.split(X=train_vecs, groups=session_list[train_index]))
                search.fit(train_vecs, train_pow_all)
                clfs = search
            else:
                for cf, arr in train_pow_all.groupby('cf'):
                    train_pow_cf = arr.squeeze()
                    search = clone(estimator)
                    if pipeline:
                        search.set_params(estimator__cv=list(cv.split(X=train_vecs, groups=session_list[train_index])))
                    else:
                        search.set_params(cv=list(cv.split(X=train_vecs, groups=session_list[train_index])))
                    search.fit(train_vecs, train_pow_cf)
    #                 print(search.best_estimator_)
    #                 best_clf = search.best_estimator_
#                     print('saving')
#                     test_fp = clf_dir + 'test.joblib'
#                     dump(search, test_fp)
                    clfs[cf] = search
                    
#             clf_fp = clf_dir + estimator_name + '_' + subject + '_' + vec_col_type + '_clf_' + str(fold_ind) + '.joblib'
#             print(clf_fp)
            dump((clfs, (train_index, test_index)), clf_fp)
            fold_ind += 1
            
def train_models_cv_perm(df_subj, perm_id, clf, word2vec_cat_df, cv, vec_col_type='item', 
                    clf_dir='perm_clfs_lolo/', preproc_dir='preproc_data/catFR1/encoding/', force=True):
    subject = df_subj.subject.iloc[0]
    clf_fp = clf_dir + subject + '_' + vec_col_type + '_clf_1.joblib'
    if os.path.exists(clf_fp) and not force:
        return True
    
    if vec_col_type == 'item':
        use_vec_cols = ['vec_' + str(i) for i in range(300)]
    else:
        use_vec_cols = ['mean_vec_' + str(i) for i in range(300)]
    
    print(clf_dir)
    z_pow_stacked_list = []
    events_df_list = []
    
    for _, df_sess in df_subj.iterrows():
        preproc_fp = preproc_dir + df_sess['subject'] + '_' + str(df_sess['session']) + '.pkl'
        if os.path.exists(preproc_fp):
            try:
                z_pow_stacked, events_df = pickle.load(open(preproc_fp, 'rb'))
                z_pow_stacked = z_pow_stacked.assign_coords(session=("event", events_df.session))
                z_pow_stacked_list.append(z_pow_stacked)
                events_df_list.append(events_df)
            except:
                print('error', df_sess['session'], sys.exc_info())
                continue
    if not len(z_pow_stacked_list) == 0:
        events_df = pd.concat(events_df_list)
        perm_df = events_df.groupby(['session', 'subject', 'list']).sample(frac=1, random_state=perm_id)
        merge_on = ['item', 'category', 'category_num']
        if df_sess['experiment'] == "FR1":
            merge_on = 'item'
        perm_df = perm_df[['session', 'subject', 'list'] + merge_on].reset_index().drop(columns='index')
        perm_df.to_csv(clf_dir + subject + '_' + vec_col_type + 'perm_' + str(perm_id) + '.csv')
        events_df = perm_df.merge(word2vec_cat_df, on=merge_on, how='left')
        z_pow_stacked = xr.concat(z_pow_stacked_list, dim='event')
        session_list = events_df['session'].astype(str) + '_' + events_df['list'].astype(str)

        fold_ind = 1
        for train_index, test_index in cv.split(events_df[use_vec_cols], groups=session_list):
            print(fold_ind)

            train_vecs = events_df.iloc[train_index].loc[:, use_vec_cols]
            train_pow_all = z_pow_stacked[train_index, :]
            print(train_vecs.shape, train_pow_all.shape)

            clf.fit(train_vecs, train_pow_all)
            clf_fp = clf_dir + subject + '_' + vec_col_type + '_clf_' + 'perm_' + str(perm_id) + '_' + str(fold_ind) + '.joblib'
            dump((clf, (train_index, test_index)), clf_fp)
            fold_ind += 1
            
def get_pred_df_lolo(word_events_df, rec_events_df, z_pow_stacked, vec_df, vec_cols,
                     clf_dir='clfs_lolo/', pred_df_dir='cat_pred_dfs_lolo/', vec_col_type='mean', 
                     item_col='category', pairs_dir='network_electrodes/', pairs_query=None, fp_prefix=''):
    pred_vecs = vec_df[vec_cols]
    list_item_df = word_events_df[['session', 'list', item_col]].drop_duplicates()
    first_recall_df = rec_events_df.groupby(['session', 'list']).first().reset_index()

    subject = word_events_df.subject.iloc[0]
    pred_dfs = []
    subj_glob = glob.glob(clf_dir + subject + '_' + vec_col_type + '*')
    
    if pairs_query != None:
        out = load(pairs_dir + subject + '_electrodes_cml.joblib') 
        pairs = out[-1]
        sel_pairs = pairs.query(pairs_query)
        sel_labels = sel_pairs['label'].values
        if fp_prefix == '':
            fp_prefix = pairs_query[0:2]
    
    for clf_fp in subj_glob:
        clf, (train_index, test_index) = load(clf_fp)
        fold_size = len(test_index)
        test_events = word_events_df.iloc[test_index]

        test_sls = test_events['session_list'].unique()
        pred_df_fp = pred_df_dir + subject + '_' + vec_col_type + '_' + fp_prefix + '_' + str(test_sls[0]) + '.csv'
        test_pow_all = z_pow_stacked.swap_dims({'event': 'session_list'}
                                                      ).sel(session_list=test_sls
                                                           ).swap_dims({'session_list': 'event'})

        Y_hat_test = clf.predict(pred_vecs)
        Y_hat_test_arr = xr.DataArray(Y_hat_test, 
                                      coords={item_col: vec_df[item_col], 
                                              'cf': z_pow_stacked.cf}, 
                                      dims=[item_col, 'cf'])
        
        if pairs_query != None:
            test_pow_all = test_pow_all.unstack('cf').sel(channel=sel_labels).stack(cf=("channel", "frequency"))
            Y_hat_test_arr = Y_hat_test_arr.unstack('cf').sel(channel=sel_labels).stack(cf=("channel", "frequency"))

        corr_arrs = []
        for z_pow_timpoint_arr in test_pow_all.transpose('time', 'event', 'cf'):
            corr_arr = xr.corr(Y_hat_test_arr, z_pow_timpoint_arr, dim='cf')
            corr_arrs.append(corr_arr)

        corr_arr = xr.concat(corr_arrs, dim='time')
        corr_df = corr_arr.to_dataframe('corr').reset_index()
        corr_df.rename(columns={'trial': 'list'}, inplace=True)

        pred_df = corr_df.merge(list_item_df, how='left', indicator=True,
                                    on=['session', 'list', item_col])
        pred_df['subject'] = subject

        pred_df = pred_df.merge(first_recall_df, 
                                on=['subject', 'session', 'list'], 
                                suffixes=('_pred', '_recall'))

        pred_df[item_col + '_prob'] = pred_df.groupby(['subject', 'session', 'list', 'time']
                                                         )['corr'].transform(sp.special.softmax)
        pred_df['log_' + item_col + '_prob'] = np.log(pred_df[item_col + '_prob'])

        pred_df[(item_col + '_type')] = 'other'
        pred_df.loc[pred_df['_merge'] == 'both', (item_col + '_type')] = 'list'
        pred_df.loc[pred_df[item_col + '_pred'] == pred_df[item_col + '_recall'], (item_col + '_type')] = 'first_recall'
        pred_df.to_csv(pred_df_fp)
    pred_dfs.append(pred_df)
    return pd.concat(pred_dfs)

def get_pred_df_lolo2(word_events_df, rec_events_df, z_pow_stacked, vec_df, vec_cols,
                     clf_dir='clfs_lolo/', pred_df_dir='cat_pred_dfs_lolo/', vec_col_type='mean', 
                     item_col='category', pairs_dir='network_electrodes/', pairs_query=None, fp_prefix='', period='distractor'):
    replay_periods = ["distractor", "last_half_encoding", "all_encoding_distractor"]
    pred_vecs = vec_df[vec_cols]
    list_item_df = word_events_df[['session', 'list', item_col]].drop_duplicates()
    first_recall_df = rec_events_df.groupby(['session', 'list']).first().reset_index()

    subject = word_events_df.subject.iloc[0]
    pred_dfs = []
    subj_glob = glob.glob(clf_dir + subject + '_' + vec_col_type + '*')
    
    if pairs_query != None:
        out = load(pairs_dir + subject + '_electrodes_cml.joblib') 
        pairs = out[-1]
        sel_pairs = pairs.query(pairs_query)
        sel_labels = sel_pairs['label'].values
        if fp_prefix == '':
            fp_prefix = pairs_query[0:2]
    
    for clf_fp in subj_glob:
        clf, (train_index, test_index) = load(clf_fp)
        fold_size = len(test_index)
        test_events = word_events_df.iloc[test_index]

        test_sls = test_events['session_list'].unique()
        pred_df_fp = pred_df_dir + subject + '_' + vec_col_type + '_' + fp_prefix + '_' + str(test_sls[0]) + '.csv'
        test_pow_all = z_pow_stacked.swap_dims({'event': 'session_list'}
                                                      ).sel(session_list=test_sls
                                                           ).swap_dims({'session_list': 'event'})

        Y_hat_test = clf.predict(pred_vecs)
        Y_hat_test_arr = xr.DataArray(Y_hat_test, 
                                      coords={item_col: vec_df[item_col], 
                                              'cf': z_pow_stacked.cf}, 
                                      dims=[item_col, 'cf'])
        
        if pairs_query != None:
            test_pow_all = test_pow_all.unstack('cf').sel(channel=sel_labels).stack(cf=("channel", "frequency"))
            Y_hat_test_arr = Y_hat_test_arr.unstack('cf').sel(channel=sel_labels).stack(cf=("channel", "frequency"))

        corr_arrs = []
        for z_pow_timpoint_arr in test_pow_all.transpose('time', 'event', 'cf'):
            corr_arr = xr.corr(Y_hat_test_arr, z_pow_timpoint_arr, dim='cf')
            corr_arrs.append(corr_arr)

        corr_arr = xr.concat(corr_arrs, dim='time')
        corr_df = corr_arr.to_dataframe('corr').reset_index()
        corr_df.rename(columns={'trial': 'list'}, inplace=True)

        pred_df = corr_df.merge(list_item_df, how='left', indicator=True,
                                    on=['session', 'list', item_col])
        pred_df['subject'] = subject

        pred_df = pred_df.merge(first_recall_df, 
                                on=['subject', 'session', 'list'], 
                                suffixes=('_pred', '_recall'))

        pred_df[item_col + '_prob'] = pred_df.groupby(['subject', 'session', 'list', 'time']
                                                         )['corr'].transform(sp.special.softmax)
        pred_df['log_' + item_col + '_prob'] = np.log(pred_df[item_col + '_prob'])

        pred_df[(item_col + '_type')] = 'other'
        pred_df.loc[pred_df['_merge'] == 'both', (item_col + '_type')] = 'list'
        pred_df.loc[pred_df[item_col + '_pred'] == pred_df[item_col + '_recall'], (item_col + '_type')] = 'first_recall'
        pred_df.to_csv(pred_df_fp)
    pred_dfs.append(pred_df)
    return pd.concat(pred_dfs)

def get_encoding_pred_dfs(events_df, z_pow_stacked, vec_df, use_vec_cols, clf_dir='clfs_lolo_new', vec_col_type='mean', item_col='category'):
    corr_dfs = []
    roc_auc_dicts = []
    subject = events_df.subject.iloc[0]
    pred_vecs = vec_df[use_vec_cols]
    
    list_item_df = events_df[['session', 'list', item_col]].drop_duplicates()
    subj_glob = glob.glob(clf_dir + subject + '_' + vec_col_type + '*')
    for list_fp in subj_glob:
        clf, (train_index, test_index) = load(list_fp)
        fold_size = len(test_index)

        test_pow_all = z_pow_stacked[test_index, :]
        test_events = events_df.iloc[test_index][['session', 'list', item_col, 'index']]
        u_items = test_events[item_col].unique()

        Y_hat_test = clf.predict(pred_vecs)

        Y_hat_test_arr = xr.DataArray(Y_hat_test, 
                                      coords={item_col: vec_df[item_col], 
                                              'cf': z_pow_stacked.cf}, 
                                      dims=[item_col, 'cf'])
        corr_arr = xr.corr(Y_hat_test_arr, test_pow_all, dim='cf')
        corr_df = corr_arr.to_dataframe('corr').reset_index()
        corr_df = corr_df.merge(test_events, 
                                left_on=['event', 'session'], 
                                right_on=['index', 'session'],
                                suffixes=('_pred', '_true'))
        corr_df = corr_df.merge(list_item_df, how='left', indicator=True,
                                left_on=['session', 'list', item_col+'_pred'], 
                                right_on=['session', 'list', item_col])
        corr_df['subject'] = subject
        heldout_session, heldout_list = corr_df[['session', 'list']].iloc[0]
        corr_df[item_col+'_prob'] = corr_df.groupby(['session', 'event']
                                                 )['corr'].transform(sp.special.softmax)
        corr_df['log_'+item_col+'_prob'] = np.log(corr_df[item_col+'_prob'])
        corr_df[item_col+'_type'] = 'other'
        corr_df.loc[corr_df[item_col+'_pred'] == corr_df[item_col], item_col+'_type'] = 'list'
        corr_df.loc[corr_df[item_col+'_pred'] == corr_df[item_col+'_true'], item_col+'_type'] = 'true'

        # Compute ROC
        pred_df = corr_df[[item_col+'_true', item_col+'_pred', 'corr', 'session', 'list', 'event']].query(
            item_col+'_pred in @u_items')
        inv_item_map = dict(enumerate(pred_df[item_col+'_true'].astype('category').cat.categories))
        item_map = {v: k for (k, v) in inv_item_map.items()}
        to_replace = {
            item_col+'_true': item_map,
            item_col+'_pred': item_map
        }
        auc_df = pred_df.replace(to_replace=to_replace).sort_values(
            ['session', 'event', item_col+'_pred'])
        auc_df[item_col+'_prob'] = auc_df.groupby(['session', 'event']
                                                 )['corr'].transform(sp.special.softmax)
        auc_df.drop(columns=['corr'], inplace=True)
        pivot_index = ['session', 'event', item_col+'_true']
        test_preds = auc_df.pivot_table(values=item_col+'_prob', columns=item_col+'_pred', 
                                            index=pivot_index)
        test_items = test_preds.reset_index()[item_col+'_true']

        roc_auc_dict = {'auc': roc_auc_score(test_items, test_preds, multi_class='ovr'),
                          'session': heldout_session,
                              'list': heldout_list,
                          'subject': subject} #ovr matches kragel i think?
#         print(roc_auc_dict)
        corr_dfs.append(corr_df)
        roc_auc_dicts.append(roc_auc_dict)
    return corr_dfs, roc_auc_dicts

def get_cat_pred_df_lolo(word_events_df, rec_events_df, z_pow_stacked, cat_vec_df, mean_vec_cols, clf_dir='clfs_lolo/', cat_pred_df_dir='cat_pred_dfs_lolo/', vec_col_type='mean'):
    pred_vecs = cat_vec_df[mean_vec_cols]
    list_cat_df = word_events_df[['session', 'list', 'category']].drop_duplicates()
    first_recall_df = rec_events_df.groupby(['session', 'list']).first().reset_index()

    subject = word_events_df.subject.iloc[0]
    cat_pred_dfs = []
    subj_glob = glob.glob(clf_dir + subject + '_' + vec_col_type + '*')
    
    for clf_fp in subj_glob:
        clf, (train_index, test_index) = load(clf_fp)
        fold_size = len(test_index)
        test_events = word_events_df.iloc[test_index]

        test_sls = test_events['session_list'].unique()
        cat_pred_df_fp = cat_pred_df_dir + subject + '_' + vec_col_type + '_' + str(test_sls[0]) + '.csv'
        test_pow_all = z_pow_stacked.swap_dims({'event': 'session_list'}
                                                      ).sel(session_list=test_sls
                                                           ).swap_dims({'session_list': 'event'})

        Y_hat_test = clf.predict(pred_vecs)
        Y_hat_test_arr = xr.DataArray(Y_hat_test, coords={'category': cat_vec_df.category, 
                                                              'cf': z_pow_stacked.cf}, dims=['category', 'cf'])

        corr_arrs = []
        for z_pow_timpoint_arr in test_pow_all.transpose('time', 'event', 'cf'):
            corr_arr = xr.corr(Y_hat_test_arr, z_pow_timpoint_arr, dim='cf')
            corr_arrs.append(corr_arr)

        corr_arr = xr.concat(corr_arrs, dim='time')
        corr_df = corr_arr.to_dataframe('corr').reset_index()
        corr_df.rename(columns={'trial': 'list'}, inplace=True)

        cat_pred_df = corr_df.merge(list_cat_df, how='left', indicator=True,
                                    on=['session', 'list', 'category'])
        cat_pred_df['subject'] = subject

        cat_pred_df = cat_pred_df.merge(first_recall_df, 
                                        on=['subject', 'session', 'list'], 
                                        suffixes=('_pred', '_recall'))

        cat_pred_df['cat_prob'] = cat_pred_df.groupby(['subject', 'session', 'list', 'time']
                                                         )['corr'].transform(sp.special.softmax)
        cat_pred_df['log_cat_prob'] = np.log(cat_pred_df['cat_prob'])

        cat_pred_df['cat_type'] = 'other'
        cat_pred_df.loc[cat_pred_df['_merge'] == 'both', 'cat_type'] = 'list'
        cat_pred_df.loc[cat_pred_df['category_pred'] == cat_pred_df['category_recall'], 'cat_type'] = 'first_recall'
        cat_pred_df.to_csv(cat_pred_df_fp)
    cat_pred_dfs.append(cat_pred_df)
    return pd.concat(cat_pred_dfs)

def get_cat_pred_df_lolo2(word_events_df, rec_events_df, z_pow_stacked, cat_vec_df, mean_vec_cols, clf_dir='clfs_lolo/', cat_pred_df_dir='cat_pred_dfs_lolo/', vec_col_type='mean', period='distractor'):
    replay_periods = ["distractor", "last_half_encoding", "all_encoding_distractor"]
    pred_vecs = cat_vec_df[mean_vec_cols]
    list_cat_df = word_events_df[['session', 'list', 'category']].drop_duplicates()
    if period in replay_periods:
        first_recall_df = rec_events_df.groupby(['session', 'list']).first().reset_index()
    
    subject = word_events_df.subject.iloc[0]
    cat_pred_dfs = []
    subj_glob = glob.glob(clf_dir + subject + '_' + vec_col_type + '*')
    
    for clf_fp in subj_glob:
        clf, (train_index, test_index) = load(clf_fp)
        fold_size = len(test_index)
        test_events = word_events_df.iloc[test_index]

        test_sls = test_events['session_list'].unique()
        cat_pred_df_fp = cat_pred_df_dir + period + '/' + subject + '_' + vec_col_type + '_' + str(test_sls[0]) + '.csv'
        if test_sls not in z_pow_stacked.session_list:
            continue
        test_pow_all = z_pow_stacked.swap_dims({'event': 'session_list'}
                                                    ).sel(session_list=test_sls[0])
        if test_pow_all.session_list.shape == ():
            test_pow_all = test_pow_all.expand_dims('event')
        else:
            test_pow_all = test_pow_all.swap_dims({'session_list': 'event'})

        Y_hat_test = clf.predict(pred_vecs)
        Y_hat_test_arr = xr.DataArray(Y_hat_test, coords={'category': cat_vec_df.category, 
                                                              'cf': z_pow_stacked.cf}, dims=['category', 'cf'])

        corr_arrs = []
        for z_pow_timpoint_arr in test_pow_all.transpose('time', 'event', 'cf'):
            corr_arr = xr.corr(Y_hat_test_arr, z_pow_timpoint_arr, dim='cf')
            corr_arrs.append(corr_arr)

        corr_arr = xr.concat(corr_arrs, dim='time')
        corr_df = corr_arr.to_dataframe('corr').reset_index()
        corr_df[['session', 'list']] = corr_df['session_list'].str.split('_', expand=True).astype(int)

        cat_pred_df = corr_df.merge(list_cat_df, how='left', indicator=True,
                                    on=['session', 'list', 'category'])
        cat_pred_df['subject'] = subject

        if period in replay_periods:
            cat_pred_df = cat_pred_df.merge(first_recall_df, 
                                        on=['subject', 'session', 'list'], 
                                        suffixes=('_pred', '_recall'))
        else:
            cat_pred_df = cat_pred_df.merge(rec_events_df, 
                      left_on=['event', 'list', 'session', 'subject'],
                      right_on=['index', 'list', 'session', 'subject'],
                      suffixes=('_pred', '_recall'), how='left')
        
        trial_group = ['subject', 'session', 'list', 'time']
        if period not in replay_periods:
            trial_group = trial_group + ['event']
        cat_pred_df['cat_prob'] = cat_pred_df.groupby(trial_group
                                                         )['corr'].transform(sp.special.softmax)
        cat_pred_df['log_cat_prob'] = np.log(cat_pred_df['cat_prob'])

        cat_pred_df['cat_type'] = 'other'
        cat_pred_df.loc[cat_pred_df['_merge'] == 'both', 'cat_type'] = 'list'
        if period in replay_periods:
            rec_label = 'first_recall'
        else:
            rec_label = 'recall'
        cat_pred_df.loc[cat_pred_df['category_pred'] == cat_pred_df['category_recall'], 'cat_type'] = rec_label 
        cat_pred_df.to_csv(cat_pred_df_fp)
    cat_pred_dfs.append(cat_pred_df)
    return pd.concat(cat_pred_dfs)

def get_cat_pred_df_kfold(word_events_df, rec_events_df, z_pow_stacked, cat_vec_df, mean_vec_cols, clf_dir='clfs_list_kfold/', cat_pred_df_dir='list_kfold_cat_pred_dfs/', force=False, n_folds=5):
    pred_vecs = cat_vec_df[mean_vec_cols]
    list_cat_df = word_events_df[['session', 'list', 'category']].drop_duplicates()
    first_recall_df = rec_events_df.groupby(['session', 'list']).first().reset_index()

    subject = word_events_df.subject.iloc[0]
    cat_pred_dfs = []
    
    for fold_ind in range(1, n_folds+1):
        cat_pred_df_fp = cat_pred_df_dir + subject + '_' + str(fold_ind) + '.csv'
        if os.path.exists(cat_pred_df_fp) and not force:
            cat_pred_df = pd.read_csv(cat_pred_df_fp)
        else:
            clf, (train_index, test_index) = load(clf_dir + subject + '_clf_' + str(fold_ind) + '.joblib')
            fold_size = len(test_index)
            test_events = word_events_df.iloc[test_index]

            test_sls = test_events['session_list'].unique()
            test_pow_all = z_pow_stacked.swap_dims({'event': 'session_list'}
                                                          ).sel(session_list=test_sls
                                                               ).swap_dims({'session_list': 'event'})

            Y_hat_test = clf.predict(pred_vecs)
            Y_hat_test_arr = xr.DataArray(Y_hat_test, coords={'category': cat_vec_df.category, 
                                                                  'cf': z_pow_stacked.cf}, dims=['category', 'cf'])

            corr_arrs = []
            for z_pow_timpoint_arr in test_pow_all.transpose('time', 'event', 'cf'):
                corr_arr = xr.corr(Y_hat_test_arr, z_pow_timpoint_arr, dim='cf')
                corr_arrs.append(corr_arr)

            corr_arr = xr.concat(corr_arrs, dim='time')
            corr_df = corr_arr.to_dataframe('corr').reset_index()
            corr_df.rename(columns={'trial': 'list'}, inplace=True)

            cat_pred_df = corr_df.merge(list_cat_df, how='left', indicator=True,
                                        on=['session', 'list', 'category'])
            cat_pred_df['subject'] = subject
            cat_pred_df['fold_ind'] = fold_ind

            cat_pred_df = cat_pred_df.merge(first_recall_df, 
                                            on=['subject', 'session', 'list'], 
                                            suffixes=('_pred', '_recall'))

            cat_pred_df['cat_prob'] = cat_pred_df.groupby(['subject', 'session', 'list', 'time']
                                                             )['corr'].transform(sp.special.softmax)
            cat_pred_df['log_cat_prob'] = np.log(cat_pred_df['cat_prob'])

            cat_pred_df['cat_type'] = 'other'
            cat_pred_df.loc[cat_pred_df['_merge'] == 'both', 'cat_type'] = 'list'
            cat_pred_df.loc[cat_pred_df['category_pred'] == cat_pred_df['category_recall'], 'cat_type'] = 'first_recall'
            cat_pred_df.to_csv(cat_pred_df_fp)
        cat_pred_dfs.append(cat_pred_df)
    return pd.concat(cat_pred_dfs)

def ClusterRun(function, parameter_list, max_cores=200):
    '''function: The routine run in parallel, which must contain all necessary
       imports internally.
    
       parameter_list: should be an iterable of elements, for which each element
       will be passed as the parameter to function for each parallel execution.
       
       max_cores: Standard Rhino cluster etiquette is to stay within 100 cores
       at a time.  Please ask for permission before using more.
       
       In jupyterlab, the number of engines reported as initially running may
       be smaller than the number actually running.  Check usage from an ssh
       terminal using:  qstat -f | egrep "$USER|node" | less
       
       Undesired running jobs can be killed by reading the JOBID at the left
       of that qstat command, then doing:  qdel JOBID
    '''
    import cluster_helper.cluster
    from pathlib import Path

    num_cores = len(parameter_list)
    num_cores = min(num_cores, max_cores)

    myhomedir = str(Path.home())
    # can add in 'mem':Num where Num is # of GB to allow for memory into extra_params
    #...Nora said it doesn't work tho and no sign it does
    # can also try increasing cores_per_job to >1, but should also reduce num_jobs to not hog
    # so like 2 and 50 instead of 1 and 100 etc. Went up to 5/20 for encoding at points
    # ...actually now went up to 10/10 which seems to stop memory errors 2020-08-12
    with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", \
        num_jobs=10, cores_per_job=20, \
        extra_params={'resources':'pename=python-round-robin'}, \
        profile=myhomedir + '/.ipython/') \
        as view:
        # 'map' applies a function to each value within an interable
        res = view.map(function, parameter_list)
        
    return res