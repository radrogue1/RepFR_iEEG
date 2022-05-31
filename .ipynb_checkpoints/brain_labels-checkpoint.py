'''
2020-02-14 JS
All regions I've been able to find in PS tasks, either in the stim. electrode (see below)
or in the other electrodes (see get_elec_regions). These can be found in a manner like
the "unique electrode region labels" cell in dataQuality.ipynb
Can import like so: >>>from brain_labels import MTL_labels, LTC_labels, PFC_labels, OTHER_labels, ALL_labels

2020-08-17 JS updated with new labels now that I'm loading localization.json pairs in addition to usual pairs in FR1.
see SWRgetRegionNames for details on getting the region names and the order or operations for differnt atlases
in SWRmodule.
2020-09-04 JS checked this for catFR too and regions are the same
2021-10-06 JS adding a general temporal lobe labels
'''


## Stein/Das labels

MTL_stein = ['left ca1','left ca2','left ca3','left dg','left sub','left prc','left ec','left phc','left mtl wm',
             'right ca1','right ca2','right ca3','right dg','right sub','right prc','right ec','right phc','right mtl wm',
             'left amy','right amy'] # including amygdala in MTL
LTC_stein = ['left middle temporal gyrus','left stg','left mtg','left itg','left inferior temporal gyrus','left superior temporal gyrus', # never saw last 2 but why not?
             'right middle temporal gyrus','right stg','right mtg','right itg','right inferior temporal gyrus','right superior temporal gyrus'] #same
PFC_stein = ['left caudal middle frontal cortex','left dlpfc','left precentral gyrus','right precentral gyrus',
             'right caudal middle frontal cortex','right dlpfc','right superior frontal gyrus']
cingulate_stein = ['left acg','left mcg','left pcg','right acg','right pcg']
parietal_stein = ['left supramarginal gyrus','right supramarginal gyrus']
other_TL_stein = ['left fusiform gyrus wm'] # actually from Das. ba36 is part of fusiform
other_stein = ['left precentral gyrus','none','right insula','right precentral gyrus','nan','misc']


# Using Desikan Neuroimage (2016), the ind localizations come from automated segmentation
# I'm also adding in dk and wb to these, since for some reason those are used for some electrode regions
# -dk comes from the same DesikanKilliany(2006) paper
# -wb (whole-brain) appears to come from FreeSurfer labels here: 
# https://www.slicer.org/wiki/Documentation/4.1/SlicerApplication/LookupTables/Freesurfer_labels
# although Sandy Das pointed to http://www.neuromorphometrics.com/2012_MICCAI_Challenge_Data.html
# 2020-08-17 updated these with new values from loading wb and MTL fields in localization.json pairs
# see https://memory-int.psych.upenn.edu/InternalWiki/index.php/RAM_data for details

MTL_ind = ['parahippocampal','entorhinal','temporalpole',   
           ' left amygdala',' left ent entorhinal area',' left hippocampus',' left phg parahippocampal gyrus',' left tmp temporal pole', # whole-brain names
           ' right amygdala',' right ent entorhinal area',' right hippocampus',' right phg parahippocampal gyrus',' right tmp temporal pole',
           'left amygdala','left ent entorhinal area','left hippocampus','left phg parahippocampal gyrus','left tmp temporal pole',
           'right amygdala','right ent entorhinal area','right hippocampus','right phg parahippocampal gyrus','right tmp temporal pole',
           '"ba35"','"ba36"','"ca1"', '"dg"', '"erc"', '"phc"', '"sub"',
           'ba35', 'ba36','ca1','dg','erc','phc','sub']
LTC_ind = ['bankssts','middletemporal','inferiortemporal','superiortemporal', # first 4 defined by Ezzyat NatComm 2018...unsure about bankssts tho
           'left itg inferior temporal gyrus','left mtg middle temporal gyrus','left stg superior temporal gyrus',
           ' left itg inferior temporal gyrus',' left mtg middle temporal gyrus',' left stg superior temporal gyrus', 
           'right itg inferior temporal gyrus','right mtg middle temporal gyrus','right stg superior temporal gyrus', 
           ' right itg inferior temporal gyrus',' right mtg middle temporal gyrus',' right stg superior temporal gyrus']
           # leaving out 'left/right ttg transverse temporal gyrus' and 'transversetemporal'
           
# localization.json (the wb ones) including TTG
PFC_ind = ['caudalmiddlefrontal','frontalpole','lateralorbitofrontal','medialorbitofrontal','parsopercularis',
          'parsorbitalis','parstriangularis','rostralmiddlefrontal','superiorfrontal']
cingulate_ind = ['caudalanteriorcingulate','isthmuscingulate','posteriorcingulate','rostralanteriorcingulate']
parietal_ind = ['inferiorparietal','postcentral','precuneus','superiorparietal','supramarginal']
occipital_ind = ['cuneus','lateraloccipital','lingual','pericalcarine']
other_TL_ind = ['fusiform','transversetemporal'] # temporal lobe but not MTL
other_ind = ['insula','none','precentral','paracentral','right inf lat vent','left inf lat vent', # not sure where to put these
            'left cerebral white matter','right cerebral white matter', # these wb labels can be anywhere in hemisphere so just put in other
             'nan','left lateral ventricle','right lateral ventricle']


## Combine across atlases

MTL_labels = MTL_stein+MTL_ind
LTC_labels = LTC_stein+LTC_ind
PFC_labels = PFC_stein+PFC_ind
OTHER_labels = cingulate_stein+parietal_stein+other_TL_stein+other_stein+ \
                cingulate_ind+occipital_ind+other_TL_ind+other_ind
ALL_labels = MTL_labels+LTC_labels+PFC_labels+OTHER_labels


## Want to create an MFG, IFG, and non-HPC MTL for Ezzyat math paper (HPC is already done below)
SFG_labels = ['left msfg superior frontal gyrus medial segment', 'left sfg superior frontal gyrus',
              'right msfg superior frontal gyrus medial segment','right sfg superior frontal gyrus',
              'right msfg superior frontal gyrus medial segment', 'right sfg superior frontal gyrus',
              'right superior frontal gyrus', 'superiorfrontal']
MFG_labels = ['left mfg middle frontal gyrus',' left mfg middle frontal gyrus',
              'right mfg middle frontal gyrus',' right mfg middle frontal gyrus']
IFG_labels = ['left opifg opercular part of the inferior frontal gyrus',' left opifg opercular part of the inferior frontal gyrus',
              'left orifg orbital part of the inferior frontal gyrus', ' left orifg orbital part of the inferior frontal gyrus',
              'left trifg triangular part of the inferior frontal gyrus',' left trifg triangular part of the inferior frontal gyrus',
              'left opifg opercular part of the inferior frontal gyrus',' left opifg opercular part of the inferior frontal gyrus',
              'left orifg orbital part of the inferior frontal gyrus', ' left orifg orbital part of the inferior frontal gyrus',
              'left trifg triangular part of the inferior frontal gyrus',' left trifg triangular part of the inferior frontal gyrus']
FG_labels = SFG_labels+MFG_labels+IFG_labels


nonHPC_MTL_labels = [MTL_labels[i] for i in range(0,len(MTL_labels)) if i not in [0,1,2,3,4,9,10,11,12,13,25,30,35,40,45,46,49,52,53,56]]
# nonHPC_MTL_labels = [MTL_labels[i] for i not in [0,1,2,3,4,9,10,11,12,13,25,30,35,40,45,46,49,52,53,56]] # all labels within MTL that aren't HPC


## what's used in SWR retrieval paper
HPC_labels = [MTL_labels[i] for i in [0,1,2,3,4,9,10,11,12,13,25,30,35,40,45,46,49,52,53,56]] # all labels within HPC
ENT_labels = [MTL_labels[i] for i in [6,15,21,24,29,34,39,47,54]] # all labels within entorhinal
PHC_labels = [MTL_labels[i] for i in [7,16,20,26,31,36,41,48,55]] # all labels within parahippocampal
AMY_labels = [MTL_labels[i] for i in [18,19,23,28,33,38]] # all labels within amygdala
ENTPHC_labels = ENT_labels+PHC_labels


# I want to select all regions in temporal lobe to create a ripple video 2021-10-06
temporal_lobe_labels = MTL_labels+LTC_labels+other_TL_stein+other_TL_ind
extra_TL = [' left fug fusiform gyrus',' right fug fusiform gyrus','left fug fusiform gyrus','right fug fusiform gyrus',
          ' left pp planum polare',' right pp planum polare','left pp planum polare','right pp planum polare',
          ' left pt planum temporale', ' right pt planum temporale','left pt planum temporale', 'right pt planum temporale',
          ' left ttg transverse temporal gyrus',' right ttg transverse temporal gyrus','left ttg transverse temporal gyrus','right ttg transverse temporal gyrus'
         ]
temporal_lobe_labels = temporal_lobe_labels+extra_TL

'''
# This is the original, which only has labels for those places STIMULATED across PS tasks.
# The above has regions for the other (record-only) electrodes as well. I dunno why you'd want to use
# this smaller set below, but keeping it for posterity
# stim location labels
# these are all the regions that were stimulated in PS and locationSearch tasks
MTL_stein = ['left ca1','left ca2','left ca3','left dg','left sub','left prc','left ec','left phc',
             'right ca1','right ca2','right ca3','right dg','right sub','right prc','right ec',
             'right phc','left mtl wm','right mtl wm','left amy','right amy'] # including amygdala in MTL
LTC_stein = ['left middle temporal gyrus','right middle temporal gyrus','right stg']
PFC_stein = ['left caudal middle frontal cortex','left dlpfc','left precentral gyrus','right precentral gyrus',
             'right caudal middle frontal cortex','right dlpfc','right superior frontal gyrus']
cingulate_stein = ['left acg','left mcg','left pcg','right acg','right pcg']
parietal_stein = ['left supramarginal gyrus','right supramarginal gyrus']
other_TL_stein = ['ba36','left fusiform gyrus wm'] # actually from Das. ba36 is part of fusiform
other_stein = ['left precentral gyrus','none','right insula','right precentral gyrus']
# Using Desikan Neuroimage (2016), the ind localizations come from automated segmentation
MTL_ind = ['parahippocampal','entorhinal','temporalpole']
LTC_ind = ['bankssts','middletemporal','inferiortemporal','superiortemporal'] # as defined by Ezzyat NatComm 2018...unsure about bankssts tho
PFC_ind = ['caudalmiddlefrontal','frontalpole','lateralorbitofrontal','medialorbitofrontal','parsopercularis',
          'parsorbitalis','parstriangularis','rostralmiddlefrontal','superiorfrontal']
cingulate_ind = ['caudalanteriorcingulate','isthmuscingulate','posteriorcingulate','rostralanteriorcingulate']
parietal_ind = ['inferiorparietal','postcentral','precuneus','superiorparietal','supramarginal']
occipital_ind = ['cuneus','lateraloccipital','lingual','pericalcarine']
other_TL_ind = ['fusiform','transversetemporal'] # temporal lobe but not MTL
other_ind = ['insula','none','precentral','paracentral'] # not sure where to put these
'''

''' Here's the whole list of reasons gotten by looping over all patients, appending regions, and uniquing:
 3rd ventricle
 left acgg anterior cingulate gyrus
 left ains anterior insula
 left amygdala
 left ang angular gyrus
 left aorg anterior orbital gyrus
 left basal forebrain
 left calc calcarine cortex
 left caudate
 left cerebellum exterior
 left cerebral white matter
 left co central operculum
 left cun cuneus
 left ent entorhinal area
 left fo frontal operculum
 left frp frontal pole
 left fug fusiform gyrus
 left gre gyrus rectus
 left hippocampus
 left inf lat vent
 left iog inferior occipital gyrus
 left itg inferior temporal gyrus
 left lateral ventricle
 left lig lingual gyrus
 left lorg lateral orbital gyrus
 left mcgg middle cingulate gyrus
 left mfc medial frontal cortex
 left mfg middle frontal gyrus
 left mog middle occipital gyrus
 left morg medial orbital gyrus
 left mpog postcentral gyrus medial segment
 left mprg precentral gyrus medial segment
 left msfg superior frontal gyrus medial segment
 left mtg middle temporal gyrus
 left ocp occipital pole
 left ofug occipital fusiform gyrus
 left opifg opercular part of the inferior frontal gyrus
 left orifg orbital part of the inferior frontal gyrus
 left pcgg posterior cingulate gyrus
 left pcu precuneus
 left phg parahippocampal gyrus
 left pins posterior insula
 left po parietal operculum
 left pog postcentral gyrus
 left porg posterior orbital gyrus
 left pp planum polare
 left prg precentral gyrus
 left pt planum temporale
 left putamen
 left sca subcallosal area
 left sfg superior frontal gyrus
 left smc supplementary motor cortex
 left smg supramarginal gyrus
 left sog superior occipital gyrus
 left spl superior parietal lobule
 left stg superior temporal gyrus
 left thalamus proper
 left tmp temporal pole
 left trifg triangular part of the inferior frontal gyrus
 left ttg transverse temporal gyrus
 left ventral dc
 right acgg anterior cingulate gyrus
 right ains anterior insula
 right amygdala
 right ang angular gyrus
 right aorg anterior orbital gyrus
 right calc calcarine cortex
 right caudate
 right cerebellum exterior
 right cerebral white matter
 right co central operculum
 right cun cuneus
 right ent entorhinal area
 right fo frontal operculum
 right frp frontal pole
 right fug fusiform gyrus
 right gre gyrus rectus
 right hippocampus
 right inf lat vent
 right iog inferior occipital gyrus
 right itg inferior temporal gyrus
 right lateral ventricle
 right lig lingual gyrus
 right lorg lateral orbital gyrus
 right mcgg middle cingulate gyrus
 right mfc medial frontal cortex
 right mfg middle frontal gyrus
 right mog middle occipital gyrus
 right morg medial orbital gyrus
 right mpog postcentral gyrus medial segment
 right mprg precentral gyrus medial segment
 right msfg superior frontal gyrus medial segment
 right mtg middle temporal gyrus
 right ocp occipital pole
 right ofug occipital fusiform gyrus
 right opifg opercular part of the inferior frontal gyrus
 right orifg orbital part of the inferior frontal gyrus
 right pcgg posterior cingulate gyrus
 right pcu precuneus
 right phg parahippocampal gyrus
 right pins posterior insula
 right po parietal operculum
 right pog postcentral gyrus
 right porg posterior orbital gyrus
 right pp planum polare
 right prg precentral gyrus
 right pt planum temporale
 right putamen
 right sca subcallosal area
 right sfg superior frontal gyrus
 right smc supplementary motor cortex
 right smg supramarginal gyrus
 right sog superior occipital gyrus
 right spl superior parietal lobule
 right stg superior temporal gyrus
 right thalamus proper
 right tmp temporal pole
 right trifg triangular part of the inferior frontal gyrus
 right ttg transverse temporal gyrus
ba35 # these had dashes before
ba36
ca1
ca3
dg
erc
misc
phc
sub
sulcus
No atlas
ba35
ba36
bankssts
ca1
caudalanteriorcingulate
caudalmiddlefrontal
cuneus
dg
entorhinal
erc
frontalpole
fusiform
inferiorparietal
inferiortemporal
insula
isthmuscingulate
lateraloccipital
lateralorbitofrontal
left acg
left acgg anterior cingulate gyrus
left ains anterior insula
left amy
left amygdala
left ang angular gyrus
left aorg anterior orbital gyrus
left ca1
left ca2
left ca3
left calc calcarine cortex
left caudal middle frontal cortex
left caudate
left cerebellum exterior
left cerebral white matter
left co central operculum
left cun cuneus
left dg
left dlpfc
left ec
left ent entorhinal area
left fo frontal operculum
left frp frontal pole
left fug fusiform gyrus
left fusiform gyrus wm
left gre gyrus rectus
left hippocampus
left inf lat vent
left iog inferior occipital gyrus
left itg inferior temporal gyrus
left lateral ventricle
left lig lingual gyrus
left lorg lateral orbital gyrus
left mcg
left mcgg middle cingulate gyrus
left mfc medial frontal cortex
left mfg middle frontal gyrus
left middle temporal gyrus
left mog middle occipital gyrus
left morg medial orbital gyrus
left mpog postcentral gyrus medial segment
left mprg precentral gyrus medial segment
left msfg superior frontal gyrus medial segment
left mtg
left mtg middle temporal gyrus
left mtl wm
left ocp occipital pole
left ofug occipital fusiform gyrus
left opifg opercular part of the inferior frontal gyrus
left orifg orbital part of the inferior frontal gyrus
left pallidum
left pcg
left pcgg posterior cingulate gyrus
left pcu precuneus
left phc
left phg parahippocampal gyrus
left pins posterior insula
left po parietal operculum
left pog postcentral gyrus
left porg posterior orbital gyrus
left pp planum polare
left prc
left precentral gyrus
left precuneus
left prg precentral gyrus
left pt planum temporale
left putamen
left sca subcallosal area
left sfg superior frontal gyrus
left smc supplementary motor cortex
left smg supramarginal gyrus
left sog superior occipital gyrus
left spl superior parietal lobule
left stg superior temporal gyrus
left sub
left supramarginal gyrus
left tc
left thalamus proper
left tmp temporal pole
left trifg triangular part of the inferior frontal gyrus
left ttg transverse temporal gyrus
left ventral dc
lingual
medialorbitofrontal
middletemporal
misc
paracentral
parahippocampal
parsopercularis
parsorbitalis
parstriangularis
pericalcarine
phc
postcentral
posteriorcingulate
precentral
precuneus
right accumbens area
right acg
right acgg anterior cingulate gyrus
right ains anterior insula
right amy
right amygdala
right ang angular gyrus
right aorg anterior orbital gyrus
right basal forebrain
right ca1
right ca2
right ca3
right calc calcarine cortex
right caudal middle frontal cortex
right caudate
right cerebellum exterior
right cerebral white matter
right co central operculum
right cun cuneus
right dg
right dlpfc
right ec
right ent entorhinal area
right fo frontal operculum
right frp frontal pole
right fug fusiform gyrus
right gre gyrus rectus
right hippocampus
right inf lat vent
right insula
right iog inferior occipital gyrus
right itg inferior temporal gyrus
right lateral ventricle
right lig lingual gyrus
right lorg lateral orbital gyrus
right mcg
right mcgg middle cingulate gyrus
right mfc medial frontal cortex
right mfg middle frontal gyrus
right middle temporal gyrus
right mog middle occipital gyrus
right morg medial orbital gyrus
right mpog postcentral gyrus medial segment
right mprg precentral gyrus medial segment
right msfg superior frontal gyrus medial segment
right mtg
right mtg middle temporal gyrus
right mtl wm
right ofug occipital fusiform gyrus
right opifg opercular part of the inferior frontal gyrus
right orifg orbital part of the inferior frontal gyrus
right pallidum
right pcg
right pcgg posterior cingulate gyrus
right pcu precuneus
right phc
right phg parahippocampal gyrus
right pins posterior insula
right po parietal operculum
right pog postcentral gyrus
right porg posterior orbital gyrus
right pp planum polare
right prc
right precentral gyrus
right prg precentral gyrus
right pt planum temporale
right putamen
right sca subcallosal area
right sfg superior frontal gyrus
right smc supplementary motor cortex
right smg supramarginal gyrus
right sog superior occipital gyrus
right spl superior parietal lobule
right stg
right stg superior temporal gyrus
right sub
right superior frontal gyrus
right supramarginal gyrus
right thalamus proper
right tmp temporal pole
right trifg triangular part of the inferior frontal gyrus
right ttg transverse temporal gyrus
rostralanteriorcingulate
rostralmiddlefrontal
sulcus
superiorfrontal
superiorparietal
superiortemporal
supramarginal
temporalpole
transversetemporal
unknown
'''