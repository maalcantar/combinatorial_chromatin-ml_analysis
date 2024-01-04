#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on November 8 15:08:15 2023
@author: alcantar
example run: 00e_nupop_occupancy_plots.py
"""
# activate virtual enviroment before running script
# source activate CombiCR

import argparse

import os
import glob
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(18,6)})
sns.set_style("white", {'axes.linewidth': 0.5})

def create_nupop_occupancy_plots():
    # create output directory for figures
    nupop_figs_dir = '../figs/nupop_results_plots/'
    # create ouput folders if needed
    CHECK_FOLDER_MAIN = os.path.isdir(nupop_figs_dir)
    if not CHECK_FOLDER_MAIN:
        os.makedirs(nupop_figs_dir)
        print(f"created folder : {nupop_figs_dir}")

    '''
    Reporter annotations

    pMAA06_102 (screen 1)
    ----------
    1-124: ZFBS
    125-376: promoter (with kozak sequence)
    377 - 1092: reporter
    1093 - 1323: terminator

    pMAA06_109 (screen 2)
    ----------
    1-65: ZFBS
    66-316: promoter (with kozak sequence)
    317-1033: reporter
    1034-1268: terminator
    1269-1333: ZFBS

    pMAA06_166 (screen 3)
    ----------
    1-124: ZFBS
    125-376: promoter (with kozak sequence)
    376 - 1092: reporter
    1093 - 1323: terminator

    pMAA06_167 (screen 4)
    ----------
    1-65: ZFBS
    66-316: promoter (with kozak sequence)
    317-1033: reporter
    1034-1268: terminator
    1269-1333: ZFBS

    pMAA06_112 (control screen)
    ----------
    1-257: promoter
    258-974: reporter
    975 - 1205: terminator
    '''

    # grab files from nupop results
    nupop_results_dir = '../data/nupop_results/'
    nupop_results_files_list = sorted(glob.glob(nupop_results_dir + '*.csv'))

    # create dictionary to annotate reporter sequences
    reporter_annotation_dict = {'pMAA06_102': ['ZFBS2']*62 + ['ZFBS1']*62 + ['promoter']*251 + ['reporter']*717 + ['terminator'] * 231,
                               'pMAA06_109': ['ZFBS1']*65 + ['promoter']*251 + ['reporter']*717 + ['terminator'] * 235 + ['ZFBS2']*65,
                               'pMAA06_166': ['ZFBS1']*62 + ['ZFBS2']*62 + ['promoter']*251 + ['reporter']*717 + ['terminator'] * 231,
                               'pMAA06_167': ['ZFBS1']*65 + ['promoter']*251 + ['reporter']*717 + ['terminator'] * 235 + ['ZFBS2']*65,
                               'pMAA06_112': ['promoter'] * 257 + ['reporter'] * 717 + ['terminator'] * 231}

    # color palette to use when plotting
    col_palette ={"flank1":"orange",
                  "ZFBS1": "blue",
                  "promoter": "purple",
                  "reporter": "green",
                  "terminator": "black",
                  "ZFBS2": "blue",
                  "flank2":"orange"}

    # analyze each reporter sequence and create a nucleosome occupancy plot
    for results_file in nupop_results_files_list:
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        fig, ax = plt.subplots()

        # load in nupop results dataframe
        current_reporter_results_df = pd.read_csv(results_file, header=0).drop('Unnamed: 0', axis=1)
        current_reporter = results_file.split('/')[-1][0:10]

        # add the sequence annotations based on the reporter type
        print(current_reporter_results_df.shape)
        print(len( reporter_annotation_dict[current_reporter]))
        current_reporter_results_df['annotation'] = reporter_annotation_dict[current_reporter]

        # create occupancy plot
        nuplot = sns.lineplot(data=current_reporter_results_df,
                     x="Position", y="Occup",
                     linewidth = 4,
                     hue='annotation',
                     palette = col_palette)
        plt.xlabel('Position', fontsize=16)
        plt.ylabel('Occupancy', fontsize=16)
        nuplot.set_yticks([0, 0.5, 1])

        # set the labels
        nuplot.set_yticklabels(['0', '0.5', '1.0'])

        output_fig_name = nupop_figs_dir + current_reporter + '_nupop_occup_plot'
        plt.savefig(output_fig_name +'.png', dpi=400)
        plt.savefig(output_fig_name +'.pdf', dpi=400)
        plt.savefig(output_fig_name +'.svg', dpi=400)

def main():
    create_nupop_occupancy_plots()

if __name__ == "__main__":
    main()
