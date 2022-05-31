import os
import numpy as np 
import pandas as pd
from copy import copy

def Loc2PairsTranslation(pairs,localizations):
    # localizations is all the possible contacts and bipolar pairs locations
    # pairs is the actual bipolar pairs recorded (plugged in to a certain montage of the localization)
    # this finds the indices that translate the localization pairs to the pairs/tal_struct

    loc_pairs = localizations.type.pairs
    loc_pairs = np.array(loc_pairs.index)
    split_pairs = [pair.upper().split('-') for pair in pairs.label] # pairs.json is usually upper anyway but things like "micro" are not
    pairs_to_loc_idxs = []
    for loc_pair in loc_pairs:
        loc_pair = [loc.upper() for loc in loc_pair] # pairs.json is always capitalized so capitalize location.pairs to match (e.g. Li was changed to an LI)
        loc_pair = list(loc_pair)
        idx = (np.where([loc_pair==split_pair for split_pair in split_pairs])[0])
        if len(idx) == 0:
            loc_pair.reverse() # check for the reverse since sometimes the electrodes are listed the other way
            idx = (np.where([loc_pair==split_pair for split_pair in split_pairs])[0])
            if len(idx) == 0:
                idx = ' '
        pairs_to_loc_idxs.extend(idx)

    return pairs_to_loc_idxs # these numbers you see are the index in PAIRS frame that the localization.pairs region will get put

def get_elec_regions(localizations,pairs): 
    # 2020-08-13 new version after consulting with Paul 
    # suggested order to use regions is: stein->das->MTL->wb->mni
    
    # 2020-08-26 previous version input tal_struct (pairs.json as a recArray). Now input pairs.json and localizations.json like this:
    # pairs = reader.load('pairs')
    # localizations = reader.load('localization')
    
    regs = []    
    atlas_type = []
    pair_number = []
    has_stein_das = 0
    
    # if localization.json exists get the names from each atlas
    if len(localizations) > 1: 
        # pairs that were recorded and possible pairs from the localization are typically not the same.
        # so need to translate the localization region names to the pairs...which I think is easiest to just do here

        # get an index for every pair in pairs
        loc_translation = Loc2PairsTranslation(pairs,localizations)
        loc_dk_names = ['' for _ in range(len(pairs))]
        loc_MTL_names = copy(loc_dk_names) 
        loc_wb_names = copy(loc_dk_names)
        for i,loc in enumerate(loc_translation):
            if loc != ' ': # set it to this when there was no localization.pairs
                if 'atlases.mtl' in localizations: # a few (like 5) of the localization.pairs don't have the MTL atlas
                    loc_MTL_names[loc] = localizations['atlases.mtl']['pairs'][i] # MTL field from pairs in localization.json
                    has_MTL = 1
                else:
                    has_MTL = 0 # so can skip in below
                loc_dk_names[loc] = localizations['atlases.dk']['pairs'][i]
                loc_wb_names[loc] = localizations['atlases.whole_brain']['pairs'][i]   
    for pair_ct in range(len(pairs)):
        try:
            pair_number.append(pair_ct) # just to keep track of what pair this was in subject
            pair_atlases = pairs.iloc[pair_ct] #tal_struct[pair_ct].atlases
            if 'stein.region' in pair_atlases: # if 'stein' in pair_atlases.dtype.names:
                test_region = str(pair_atlases['stein.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region != 'nan'):
                    regs.append(test_region.lower())
#             if 'stein' in pair_atlases.dtype.names:  ### OLD WAY FROM TAL_STRUCT...leaving as example
#                 if (pair_atlases['stein']['region'] is not None) and (len(pair_atlases['stein']['region'])>1) and \
#                    (pair_atlases['stein']['region'] not in 'None') and (pair_atlases['stein']['region'] != 'nan'):
#                     regs.append(pair_atlases['stein']['region'].lower())
                    atlas_type.append('stein')
                    has_stein_das = 1 # temporary thing just to see where stein/das stopped annotating
                    continue # back to top of for loop
                else:
                    pass # keep going in loop
            if 'das.region' in pair_atlases:
                test_region = str(pair_atlases['das.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region != 'nan'):
                    regs.append(test_region.lower())
                    atlas_type.append('das')
                    has_stein_das = 1
                    continue
                else:
                    pass
            if len(localizations) > 1 and has_MTL==1:             # 'MTL' from localization.json
                if loc_MTL_names[pair_ct] != '' and loc_MTL_names[pair_ct] != ' ':
                    if str(loc_MTL_names[pair_ct]) != 'nan': # looking for "MTL" field in localizations.json
                        regs.append(loc_MTL_names[pair_ct].lower())
                        atlas_type.append('MTL_localization')
                        continue
                    else:
                        pass
                else:
                    pass
            if len(localizations) > 1:             # 'whole_brain' from localization.json
                if loc_wb_names[pair_ct] != '' and loc_wb_names[pair_ct] != ' ':
                    if str(loc_wb_names[pair_ct]) != 'nan': # looking for "MTL" field in localizations.json
                        regs.append(loc_wb_names[pair_ct].lower())
                        atlas_type.append('wb_localization')
                        continue
                    else:
                        pass
                else:
                    pass
            if 'wb.region' in pair_atlases:
                test_region = str(pair_atlases['wb.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region != 'nan'):
                    regs.append(test_region.lower())
                    atlas_type.append('wb')
                    continue
                else:
                    pass
            if len(localizations) > 1:             # 'dk' from localization.json
                if loc_dk_names[pair_ct] != '' and loc_dk_names[pair_ct] != ' ':
                    if str(loc_dk_names[pair_ct]) != 'nan': # looking for "dk" field in localizations.json
                        regs.append(loc_dk_names[pair_ct].lower())
                        atlas_type.append('dk_localization')
                        continue
                    else:
                        pass
                else:
                    pass
            if 'dk.region' in pair_atlases:
                test_region = str(pair_atlases['dk.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region != 'nan'):
                    regs.append(test_region.lower())
                    atlas_type.append('dk')
                    continue
                else:
                    pass
            if 'ind.corrected.region' in pair_atlases: # I don't think this ever has a region label but just in case
                test_region = str(pair_atlases['ind.corrected.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region not in 'nan'):
                    regs.append(test_region.lower())
                    atlas_type.append('ind.corrected')
                    continue
                else:
                    pass  
            if 'ind.region' in pair_atlases:
                test_region = str(pair_atlases['ind.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region != 'nan'):
                    regs.append(test_region.lower())
                    atlas_type.append('ind')
                    # [tal_struct[i].atlases.ind.region for i in range(len(tal_struct))] # if you want to see ind atlases for comparison to above
                    # have to run this first though to work in ipdb: globals().update(locals())                  
                    continue
                else:
                    regs.append('No atlas')
                    atlas_type.append('No atlas')
            else: 
                regs.append('No atlas')
                atlas_type.append('No atlas')
        except AttributeError:
            regs.append('error')
            atlas_type.append('error')
    return np.array(regs),np.array(atlas_type),np.array(pair_number),has_stein_das