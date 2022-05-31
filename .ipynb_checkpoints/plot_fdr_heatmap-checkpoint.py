'''
FDR CORRECTION PLOTTING CODE TO PLOT BLACK OUTLINES OF SIGNIFICANT REGIONS 
'''
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 200)
import statsmodels.api as sm

from matplotlib import pyplot as plt

import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
import seaborn as sns
sns.set(color_codes=True)

print("plotting imports imported")

def fdr_heatmap(t, p, title, x_label, y_label, freqs, region = False, **kws):
    
    cmap = kws.get('cmap', 'RdBu_r')
    figsize = kws.get('figsize', (8,4))
    vmin = kws.get("vim", -3)
    vmax = kws.get("vmax", 3)
    rel_start = kws.get("rel_start", 0)
    rel_stop = kws.get("rel_stop", 1600)
    xtick_step = kws.get('xtick_step', 200) # ms

    
    plt.close()
    fig, ax = plt.subplots(figsize = figsize, dpi = 600)
    font = {'tick': 7, 'annot': 14, 'label': 12, 'fig':10}
    ax = np.ravel(ax)
    iax = 0
    _ax = ax[iax]
    
    valuematrix = t
    reject, pval_corrected, asid, abon = sm.stats.multipletests(np.asarray(p).flatten(), alpha = 0.05, method = 'fdr_tsbky')
    sigmatrix = reject.reshape(p.shape)
    
    vticks = np.round(np.linspace(np.ceil(vmin), np.floor(vmax), 5), 1)
    spine_len = 2
    spine_lw = 0.5
    cbar_label = '$t$-statistic'
    
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
    
    
    if region == True:
        #hard-coding bad >:(
        modified_ROI_order = ['LAI', 'RAI', 'LAS', 'RAS', 'RPI', 'RPS']
        rois_orig_order = ['LAI', 'RAI', 'LAS', 'RAS', 'LPI', 'RPI', 'LPS', 'RPS']    
        roi_idx = [rois_orig_order.index(roi) for roi in modified_ROI_order[:6]]
        
        roi_list = ['LAS','LAI','LPS','LPI','RAS','RAI','RPS','RPI']
        xticks = np.arange(len(roi_list))
        xticklabels = roi_list
        yticks = np.arange(len(freqs))[::4]
        yticklabels = freqs[::4].astype(int)
        labelpad = 4
        
        plot = _ax.imshow(sig_region.T[:,roi_idx ], origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plot = _ax.imshow(nonsig_region.T[:,roi_idx], origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        
        _ax.set_xticks(xticks)
        _ax.set_xticklabels(xticklabels)
        _ax.set_xlabel(x_label, fontsize=font['label'], labelpad=labelpad)
        plot_pixel_contour(_ax, sigmatrix.T[:, roi_idx])

        
    else: 
        dur_ms = rel_stop - rel_start 
        n_samp = t.shape[1]
        
        t0 = np.round(n_samp * (-rel_start / dur_ms), 0).astype(int) # vocalization onset (samples)
        n_ticks = int((rel_stop - rel_start) / xtick_step) + 1
        xticks = np.linspace(0, n_samp, n_ticks).astype(int)
        xticklabels = np.linspace(rel_start, rel_stop, n_ticks).astype(int)
        yticks = np.arange(len(freqs))[::4]
        yticklabels = freqs[::4].astype(int)
        labelpad = 4
        
        plot = _ax.imshow(sig_region, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plot = _ax.imshow(nonsig_region, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plot_pixel_contour(_ax, sigmatrix)
        
    _ax.set_yticks(yticks)
    _ax.set_yticklabels(yticklabels)
    _ax.set_ylabel(y_label, fontsize=font['label'], labelpad = labelpad)
    _ax.grid(False)


    plt.colorbar(plot)
    plt.title(title)


