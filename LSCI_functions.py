import pandas as pd
import os
from natsort import natsorted
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy import sparse

__author__ = 'Elena Maria Daniela Hindinger'


def saver(fig, savename):
    formats = ['svg', 'pdf']
    for fileformat in formats:
        temp_savename = os.path.join(savename + '.%s' % fileformat)
        fig.savefig(temp_savename, format=fileformat, bbox_inches='tight', dpi=300)


def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)


def subplot_figure(df):
    fig, axes = plt.subplots(7, 1, sharey=True, figsize=(10, 10))
    adjustFigAspect(fig, aspect=0.8)
    plt.suptitle('Average perfusion traces over time', fontsize=36)
    sns.despine(bottom=True, right=True, top=True)
    colors = ['b', 'g', 'orange', 'r', 'magenta', 'navy', 'teal']
    ylabel = ['R frontal', 'L frontal', 'R parietal', 'L parietal', 'occipital', 'bregma', 'lambda']
    plt.subplots_adjust(hspace=-0.02)
    for ax in np.arange(7):
        axes[ax].plot(np.arange(frames), df[df.columns[ax]], color=colors[ax])
        axes[ax].set_ylabel(ylabel[ax], fontsize=32, rotation='horizontal', labelpad=110)
        axes[ax].set_xticks(np.arange(0))
        axes[ax].set_yticks(np.arange(0))
    # plt.xticks(np.arange(0, (frames + 1), 120), np.arange(minutes + 1))
    plt.tick_params(labelsize=32)
    axes[6].set_xlabel('Time (min)', fontsize=42, labelpad=50)
    savename = os.path.join(folder_path, '%s_fluorescence_traces' % name)
    saver(fig, savename)
    plt.close('all')


def traces(df, mode):
    print 'Drawing traces'
    sns.set_style('white')
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    plt.suptitle('ROI-specific perfusion traces over time', fontsize=36)
    #sns.despine(bottom=True, right=True, top=True)
    colors = ['b', 'g', 'orange', 'r', 'magenta', 'navy', 'teal']
    ylabel = ['R frontal', 'L frontal', 'R parietal', 'L parietal', 'occipital', 'bregma', 'lambda']
    for plot in np.arange(7):
        ax.plot(np.arange(frames), df[df.columns[plot]], color=colors[plot], label=ylabel[plot])
    ax.set_xlabel('Time (min)', fontsize=20, labelpad=30)
    ax.set_ylabel('Perfusion Units', fontsize=20, labelpad=30)
    plt.xticks(np.arange(0, (frames + 1), 300), np.arange(0, 120, 10))
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(600))
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    plt.legend()
    savename = os.path.join(results_folder, '%s_perfusion_traces_%s' % (name, mode))
    saver(fig, savename)
    plt.close('all')

def fixed_baseline(y, lam=100000000, p=0.005, niter=10):
    """Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm
    Y is one trace as a np array
    (P. Eilers, H. Boelens 2005)
    CODE FROM http://stackoverflow.com/questions/29156532/python-baseline-correction-library - Paper:ALS_Linda
    """
    if np.shape(y)[0] > np.shape(y)[1]:
        y = y.T
    elif len(np.shape(y)) > 1:
        y = y[0]
    else:
        pass
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return y-z, z