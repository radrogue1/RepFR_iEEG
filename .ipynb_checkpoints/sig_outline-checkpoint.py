'''
black border contouring for significant regions 

Created: 4/20/22 by Brandon Katerman 

'''

import sys
import csv 
import os 
import os.path as op 


import numpy as np 

import pandas as pd 
pd.set_option("display.max_columns", 200)

import statsmodels.api as sm
import xarray as xr

import scipy.stats as stats 
import scipy.spatial as spatial 
#from sklearn import preprocessing

#plotting 
from matplotlib import pyplot as plt

import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
import seaborn as sns; sns.set(color_codes=True)
sns.set_style(style = 'white')

print('modules imported')
def fdr_heatmap(tvals, pvals, figsize, cmap, vmin, vmax, title, x_label, y_labels, freqs, best_regions, analysis, _ax = None):
    tvals = tvals.T
    pvals = pvals.T

    t = xr.DataArray(tvals, dims = ['regions', 'freqs'],
                   coords = {'regions': best_regions})
    p = xr.DataArray(pvals, dims = ['regions', 'freqs'],
                   coords = {'regions': best_regions})
    
    
    plt.close()
    fig, ax = plt.subplots(figsize = figsize, dpi = 600 )
    font = {'tick': 7, 'annot': 10, 'label': 8, 'fig': 10}
    ax = np.ravel(ax)
    iax = 0
    _ax = ax[iax]
    
    valuematrix = t
    reject, pval_corrected, asid, abon = sm.stats.multipletests(np.asarray(pvals).flatten(), alpha = 0.05, method = 'fdr_tsbky')
    sigmatrix = reject.reshape(pvals.shape)

    #fig, ax = plt.subplots(figsize = figsize )
    cmap =cmap 
    vticks = np.round(np.linspace(np.ceil(vmin), np.floor(vmax), 5), 1)
    spine_len = 2
    spine_lw = 0.5
    cbar_label = '$t$-value'
    
    def plot_pixel_contour(ax, sigmatrix):
            f = lambda x,y: sigmatrix[int(y),int(x)]
            g = np.vectorize(f)

            x = np.linspace(0, sigmatrix.shape[1], sigmatrix.shape[1]*100)
            y = np.linspace(0, sigmatrix.shape[0], sigmatrix.shape[0]*100)
            X, Y= np.meshgrid(x[:-1],y[:-1])
            Z = g(X[:-1],Y[:-1])

            ax.contour(Z[::-1], [0.5], colors='k', linewidths=[2], origin='upper',
                       extent=[0-0.5, x[:-1].max()-0.5,0-0.5, y[:-1].max()-0.5])
                        #extent=[0-0.5, x[:-1].max()-0.5,0-0.5, y[:-1].max()-0.5])

    sig_region = np.ma.masked_where(~sigmatrix, valuematrix)
    nonsig_region = np.ma.masked_where(sigmatrix, valuematrix)


    plot = _ax.imshow(sig_region.T[:,: ], origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plot = _ax.imshow(nonsig_region.T[:,:], origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        
    ytick = np.arange(len(freqs))[::4] 
    labelpad = 4
    plt.xticks(range(len(best_regions)),best_regions, fontsize = 12, rotation = 45)
    _ax.set_yticks(ytick)
    _ax.set_yticklabels(freqs[::4].astype(int))
    _ax.set_ylabel("Frequency (Hz)", fontsize = font['label'], labelpad = labelpad)
    
    plt.colorbar(plot)
   
    _ax.grid(False)
    
    
#     plt.title(title)

    plot_pixel_contour(_ax, sigmatrix.T[:,:])
    plt.savefig('{}.pdf'.format(analysis), bbox_inches='tight')