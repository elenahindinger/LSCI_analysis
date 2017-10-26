import pandas as pd
import os
from natsort import natsorted
from matplotlib import pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from LSCI_functions import saver, fixed_baseline, adjustFigAspect

__author__ = 'Elena Maria Daniela Hindinger'

folder_path = r'I:\Laser Speckle data\ROI raw data'
#folder_path = r'/Users/elenahindinger/Documents/FINALYEAR/HonoursProject/ROI raw data'
print natsorted(os.listdir(folder_path))
for i in natsorted(os.listdir(folder_path)):
    if i.endswith('xlsx'):
        file1 = os.path.join(folder_path, i)
        print 'working on ', file1

    else:
        pass
results_folder = os.path.join(folder_path, 'results')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
experiment_names = ['5mm2_circles', '3mm2_circles', '1mm2_circles', 'differential_shapes', 'refined_shapes']
ROI_names = ['R frontal', 'L frontal', 'R parietal', 'L parietal', 'occipital', 'bregma', 'lambda']
modes = ['raw', 'minmax_scaled', 'standardised', 'differenced', 'flattened_baseline']

for sheetnumber in np.arange(5):
    print 'Processing sheet number: ', sheetnumber
    whole_sheet = pd.read_excel(file1, sheetname=sheetnumber)
    experiment_name = experiment_names[sheetnumber]
    df = whole_sheet.drop(whole_sheet.columns[[0, 1, 9, 10, 11]], axis=1)
    df.columns = ROI_names
    frames = df.shape[0]
    raw_df = df.copy()
    #cmat = data_only.corr()
    #plt.imshow(cmat, interpolation='nearest', cmap='coolwarm', vmin=-1, vmax=1)

    ''' This section uses the min max scaler to normalise data '''
    minmaxscaler = MinMaxScaler(feature_range=(0, 1))
    minmaxscaler = minmaxscaler.fit(df)
    minmaxscaled = minmaxscaler.transform(df)
    minmaxscaled_df = pd.DataFrame(minmaxscaled, columns=ROI_names)

    ''' This section uses the standard scaler to standardise (center mean, std) data '''
    standardscaler = StandardScaler()
    standardscaler = standardscaler.fit(df)
    standardscaled = standardscaler.transform(df)
    standardscaled_df = pd.DataFrame(standardscaled, columns=ROI_names)

    ''' This section differences the data '''
    differenced_df = pd.DataFrame()
    for name, column in df.iteritems():
        differenced = column.diff()
        as_df = pd.DataFrame(differenced)
        as_df.columns = [name]
        differenced_df = pd.concat([differenced_df, as_df], axis=1)

    ''' This section flattens the baseline '''
    flattened_df = pd.DataFrame()
    for name, column in df.iteritems():
        temp = fixed_baseline(np.array([column.values]))[0]
        #baseline = np.percentile(temp, 10, axis=0)
        #dff = (column - baseline) / (baseline+0.1)
        as_df = pd.DataFrame(temp)
        as_df.columns = [name]
        flattened_df = pd.concat([flattened_df, as_df], axis=1)

    dfs = [raw_df, minmaxscaled_df, standardscaled_df, differenced_df, flattened_df]


    def traces(df, mode):
        ''' This function takes a dataframe consisting of 7 columns, and draws them all in the same plot. It will be
        saved as a pdf and svg in the results folder. '''
        print 'Drawing traces for', mode
        sns.set_style('white')
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        plt.suptitle('ROI-specific perfusion traces over time', fontsize=36)
        #sns.despine(bottom=True, right=True, top=True)
        colors = ['b', 'g', 'orange', 'r', 'magenta', 'navy', 'teal']
        ylabel = ROI_names
        for plot in np.arange(7):
            ax.plot(np.arange(frames), df[df.columns[plot]], color=colors[plot], label=ylabel[plot])
        ax.set_xlabel('Time (min)', fontsize=20, labelpad=30)
        ax.set_ylabel('Perfusion Units', fontsize=20, labelpad=30)
        plt.xticks(np.arange(0, (frames + 1), 300), np.arange(0, 120, 10))
        lg = ax.legend(loc=5, fontsize='x-large', bbox_to_anchor=(1.15, 0.6),
                       borderaxespad=0, title=mode, frameon=True)
        plt.setp(lg.get_title(), fontsize='xx-large')
        savename = os.path.join(results_folder, '%s_perfusion_traces_%s' % (experiment_name, mode))
        saver(fig, savename)
        plt.close('all')


    def subplot_figure(df, mode, yax=True):
        fig, axes = plt.subplots(7, 1, sharey=yax, sharex=True, figsize=(20, 10))
        #adjustFigAspect(fig, aspect=0.8)
        plt.suptitle('ROI-specific perfusion traces over time', fontsize=36)
        sns.despine(bottom=True, right=True, top=True)
        colors = ['b', 'g', 'orange', 'r', 'magenta', 'navy', 'teal']
        ylabel = ROI_names
        plt.subplots_adjust(hspace=0.2)
        for ax in np.arange(7):
            axes[ax].plot(np.arange(frames), df[df.columns[ax]], color=colors[ax])
            axes[ax].set_ylabel(ylabel[ax], fontsize=20, rotation='horizontal', labelpad=60)
        axes[6].set_xlabel('Time (min)', fontsize=20, labelpad=20)
        plt.xticks(np.arange(0, (frames + 1), 300), np.arange(0, 120, 10))
        savename = os.path.join(results_folder, '%s_separate_perfusion_traces_sharey_%s_%s' % (experiment_name, yax, mode))
        saver(fig, savename)
        plt.close('all')


    def cluster_map(df, mode):
        print 'Calculating cluster map for ', mode
        df.columns = ROI_names
        colors = ['b', 'g', 'orange', 'r', 'magenta', 'navy', 'teal']
        network_lut = dict(zip(map(str, df.columns), colors))
        networks = df.columns
        network_colors = pd.Series(networks, index=df.columns).map(network_lut)
        sns.set()
        sns.set_style('white')

        cg = sns.clustermap(df.corr(), cmap="BrBG",
                            row_colors=network_colors, col_colors=network_colors, figsize=(20, 20), vmin=-1, vmax=1)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        # plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=90)
        plt.setp(cg.ax_heatmap.tick_params(labelsize=20))
        savename = os.path.join(results_folder, '%s_clustermap_%s' % (experiment_name, mode))
        formats = ['pdf']
        for fileformat in formats:
            temp_savename = os.path.join(savename + '.%s' % fileformat)
            plt.savefig(temp_savename, format=fileformat, bbox_inches='tight', dpi=300)
        plt.close('all')

    for i in np.arange(len(dfs)):
        traces(df=dfs[i], mode=modes[i])
        subplot_figure(df=dfs[i], mode=modes[i])
        subplot_figure(df=dfs[i], mode=modes[i], yax=False)
        cluster_map(df=dfs[i], mode=modes[i])
    print 'Done, next one now.'
print 'All done !'
