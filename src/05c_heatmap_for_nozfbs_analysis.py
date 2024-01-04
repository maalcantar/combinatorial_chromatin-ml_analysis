#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on November 14 20:41:14 2023
@author: alcantar
example run: python 05c_heatmap_for_nozfbs_analysis.py -i ../../combicr_data/combicr_outputs/maa-005_nozfbs/flow_seq_results/ -r pMAA112 -b 6
"""
# activate virtual enviroment before running script
# source activate CombiCR

import argparse

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import collections

import seaborn as sns

def create_control_clustered_flow_seq_score_heatmap(results_merged_df, cr_order,
                                                    out_path, save_heatmap=False):

    '''
    create heatmaps containing flow seq scores for all CR interactions. each CR
    in the heatmpap is clustered by chromatin regulating complex or protein class.
    this function is heavily adapted from 'create_flow_seq_score_heatmap' from
    script 05_merge_replicates_create_heatmap.py

    PARAMETERS
    --------------------
    results_merged_df: pandas dataframe
        dataframe with averaged, normalized flow seq data for all CR interactions
    cr_order: list of strs
        order in which to plot CRs in the final heatmap (clustered by annotation)
    out_path: str
        path to output folder

    RETURNS
    --------------------
    NONE: just creates plots

    '''

    # create lists with CRs that appear in the first and second position
    regulators_first_position = []
    regulators_second_position = []
    for CR_combos in list(results_merged_df.index):
        CR_combos_split_temp = CR_combos.split('_')

        CR_position1_temp = CR_combos_split_temp[0]
        CR_position2_temp = CR_combos_split_temp[1]

        if (CR_position1_temp not in regulators_first_position):
            regulators_first_position.append(CR_position1_temp)
        if (CR_position2_temp not in regulators_second_position):
            regulators_second_position.append(CR_position2_temp)

    # define order for CRs
    regulators_first_position = [CR for CR in cr_order if CR in regulators_first_position]
    regulators_second_position = [CR for CR in cr_order if CR in regulators_second_position]
    # initialize empty dataframe that will contain flow seq scores for all
    # combinations
    CR_combos_interactions = pd.DataFrame(columns= regulators_second_position, index=regulators_first_position)
    flow_seq_scores = results_merged_df['flow_seq_score_on-off']

    for CR_combos in list(results_merged_df.index):
        CR_combos_split_temp = CR_combos.split('_')

        CR_position1_temp = CR_combos_split_temp[0]
        CR_position2_temp = CR_combos_split_temp[1]

        CR_combos_interactions.at[CR_position1_temp,CR_position2_temp] = flow_seq_scores[CR_combos]
    CR_combos_interactions = CR_combos_interactions.fillna(np.nan)

    CR_combos_interactions.rename(index={'dpd4': 'dpb4', 'hmt2': 'hmt1'},
                             columns={'dpd4': 'dpb4', 'hmt2': 'hmt1'})

    CR_combos_interactions.index = CR_combos_interactions.index.str.upper()
    CR_combos_interactions.columns = CR_combos_interactions.columns.str.upper()

    # create heatmap with flow seq score for all combinations
    plt.figure(figsize=(20,20))
    plt.rcParams.update({'font.size': 18,
                         'font.family':'arial'})

    # make combinations that did not appear in the screen black
    # note that matplotlib v3.3.1 will make sure top and bottom rows are not
    # clipped. some other versions of matplotlib will cause this clipping issue
    # (i.e., https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot)
    heatmap = sns.heatmap(CR_combos_interactions, cmap="Greens", xticklabels=True,
                yticklabels=True, mask=CR_combos_interactions.isnull(), square =True)
    heatmap.set_facecolor("black")

    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.ylabel('First position', fontsize=18)
    plt.xlabel('Second position', fontsize=18)

    if save_heatmap:
        plt.savefig(out_path+'.png', dpi=400)
        plt.savefig(out_path+'.pdf', dpi=400)
        plt.savefig(out_path+'.svg', dpi=400)

    # remove nan values that emerge from subtracting on and off dataframes
    # (# some combinations have one value but not the other which creates
    # an nan value)
    results_merged_dropna_df = results_merged_df.dropna()

     # plot bliss scores
    plt.rcParams.update({'font.size': 18,
                         'font.family':'sans-serif'})

    fig, axes = plt.subplots(2, 1,  gridspec_kw={"height_ratios":(.30, .10)}, figsize = (13, 10))

    histogram = sns.histplot(x=results_merged_dropna_df['flow_seq_score_on-off'], color='cornflowerblue', ax=axes[0])
    boxplot = sns.boxplot(x=results_merged_dropna_df['flow_seq_score_on-off'], color='cornflowerblue', ax=axes[1])

    if save_heatmap:
        plt.savefig(out_path+'_histogram.png', dpi=400)
        plt.savefig(out_path+'_histogram.pdf', dpi=400)
        plt.savefig(out_path+'_histogram.svg', dpi=400)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to flow_seq_results directory')
    parser.add_argument('-r', nargs='*', help='reporter plasmids in dataset',required=True)
    parser.add_argument('-b', help='number of bins', default=8)

    args = parser.parse_args()

    results_df_path_prefix = args.i
    reporter_plasmids = args.r #
    num_bins = int(args.b)#6

    if results_df_path_prefix[-1] != '/':
        results_df_path_prefix = results_df_path_prefix + '/'

    out_dir = results_df_path_prefix.replace('flow_seq_results', 'flow_seq_score_heatmaps')
    CHECK_FOLDER_MAIN = os.path.isdir(out_dir)
    if not CHECK_FOLDER_MAIN:
        os.makedirs(out_dir)
        print(f"created folder : {out_dir}")

    merged_df_off_on_list = []

    # this handles the case where there are multiple control reporters
    # in our experiments, we only had one, so the loop runs once.
    for reporter in reporter_plasmids:
        results_off_df_path = results_df_path_prefix + 'pMAA06_' + reporter + '_off_flow-seq_results.csv'
        results_on_df_path = results_df_path_prefix + 'pMAA06_' + reporter + '_on_flow-seq_results.csv'

    off_df = pd.read_csv(results_off_df_path, index_col=0)
    on_df = pd.read_csv(results_on_df_path, index_col=0)
    fc_rep1_df = pd.DataFrame()
    fc_rep2_df = pd.DataFrame()

    # compute ON-OFF values for paired replicates (i.e.,ON_rep1 - OFF rep1)
    fc_rep1_df['on-off_rep1'] = on_df['flow_seq_score_rep_1'] - off_df['flow_seq_score_rep_1']
    fc_rep2_df['on-off_rep2'] = on_df['flow_seq_score_rep_2'] - off_df['flow_seq_score_rep_2']
    fc_all_df = pd.merge(fc_rep1_df, fc_rep2_df, left_index=True, right_index=True)

    # average the two ON-OFF values
    fc_all_df['flow_seq_score_on-off'] = fc_all_df.mean(axis=1)

    out_path = out_dir + 'pMAA06_' + reporter.replace('_on','') + '_flow-seq_scores_on-off'

    # define all CRs by their chromatin regulating complex or protein class
    acetyltransferase_complex = ['hat1', 'hat2']
    deacetyltransferase_complex = ['hda1', 'hos1', 'hos2', 'hst2', 'sir2']
    rsc = ['htl1', 'ldb7', 'sfh1']
    swr1 = ['arp6', 'swc5', 'vps71']
    compass = ['bre2', 'spp1', 'swd3']
    mediator = ['med4', 'med7', 'nut2', 'srb7']
    saga = ['ada2', 'gcn5', 'sus1']
    kinase = ['gapdh', 'pyk1', 'pyk2', 'tdh3']
    methyltransferase = ['DAM', 'hmt2']
    swi_snf = ['arp9', 'rtt102', 'snf11']
    other = ['btt1', 'cac2', 'cdc36', 'dpd4', 'eaf3', 'eaf5', 'esa1', 'hhf2',
             'ies5','lge1', 'mig1', 'nhp6a','nhp10',  'rpd3', 'vp16', 'empty']

    CR_list_all = acetyltransferase_complex + compass + kinase + \
                    deacetyltransferase_complex + mediator + methyltransferase + \
                    rsc + saga + swi_snf + swr1 + other

    create_control_clustered_flow_seq_score_heatmap(fc_all_df, CR_list_all,
                                            out_path, save_heatmap=True)

if __name__ == "__main__":
    main()
