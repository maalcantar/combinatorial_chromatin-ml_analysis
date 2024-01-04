#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on November 11 16:15:56 2023
@author: alcantar
example run: python 05b_cluster_heatmap.py -i ../../combicr_data/combicr_outputs/maa-004/flow_seq_results/pMAA06_pMAA102_flow-seq_results_final.csv -s 1
"""
# activate virtual enviroment before running script
# source activate CombiCR

import argparse

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def create_clustered_flow_seq_score_heatmap(results_merged_df, cr_order, out_path, save_heatmap=0):

    '''
    create heatmaps containing flow seq scores for all CR interactions. each CR
    in the heatmpa is clustered by chromatin regulating complex or protein class.
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
    flow_seq_scores = results_merged_df['flow_seq_score']

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to flow_seq_results merged csv')
    parser.add_argument('-s', help='save heatmap?', default=0)

    args = parser.parse_args()

    results_merged_path = args.i
    save_heatmap = int(args.s)

    results_merged_df = pd.read_csv(results_merged_path, index_col=0)
    out_path = results_merged_path.replace('flow_seq_results', 'flow_seq_score_heatmaps')
    out_path = out_path.replace('_results_final.csv', '_scores_clustered')

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
                    rsc + saga +swi_snf + swr1 + other

    create_clustered_flow_seq_score_heatmap(results_merged_df, CR_list_all,
                                            out_path, save_heatmap)

if __name__ == "__main__":
    main()
