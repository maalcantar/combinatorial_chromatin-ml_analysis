#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on October 20 15:43:20 2022
@author: alcantar
example run: python 05_merge_replicates_create_heatmap.py -i ../../combicr_data/combicr_outputs/maa-002_test/flow_seq_results/ -r pMAA102 -b 4
example run 2: python 05_merge_replicates_create_heatmap.py -i ../../combicr_data/combicr_outputs/maa-004/flow_seq_results/ -r pMAA102 pMAA109 -b 8
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

def invert_dictionary(dict_to_invert):

    '''
    invert dictionary: that is, keys become values and vice-versa.
    adapted from:
    https://stackoverflow.com/questions/8377998/swapping-items-in-a-dictionary-and-order

    PARAMETERS
    --------------------
    dict_to_invert: dict
        dictionary to be inverted

    RETURNS
    --------------------
    inverted_dictionary: dict
        dictionary that has been inverted
    '''

    inverted_dictionary = collections.OrderedDict((dict_to_invert[key_val], \
    key_val) for key_val in sorted(dict_to_invert))

    return(inverted_dictionary)

def merge_replicates(results_df_path, num_bins):

    '''
    merges normalized bin counts and flow seq score for two biological replicates
    (important to note that this script assumes two replicates were taken per screen/
    reporter)

    PARAMETERS
    --------------------
    dict_to_invert: dict
        dictionary to be inverted

    RETURNS
    --------------------
    inverted_dictionary: dict
        dictionary that has been inverted


    '''
    results_df = pd.read_csv(results_df_path, index_col=0)
    index_to_col_names_dict = dict(enumerate(list(results_df.columns),1))
    col_name_to_index_dict = invert_dictionary(index_to_col_names_dict)


    flow_seq_score_rep1_name = 'flow_seq_score_rep_1'
    flow_seq_score_rep2_name = 'flow_seq_score_rep_2'
    cr_combo_merged_list = []
    # for each CR combination, average each normalized bin count and the final
    # flow seq score
    for cr_combo in results_df.itertuples():
        cr_combo_means_tmp_dict = dict()
        for bin_num in list(range(1,num_bins+1)):
            bin_name = 'bin' + str(bin_num)
            bin_name_rep1 = bin_name +'_rep1'
            bin_name_rep2 = bin_name +'_rep2'
            norm_bin_count_rep1 = cr_combo[int(col_name_to_index_dict[bin_name_rep1])]
            norm_bin_count_rep2 = cr_combo[int(col_name_to_index_dict[bin_name_rep2])]
            mean_norm_bin_count = np.mean([norm_bin_count_rep1, norm_bin_count_rep2])

            cr_combo_means_tmp_dict.update({bin_name: mean_norm_bin_count})

        flow_seq_score_rep1_tmp = cr_combo[int(col_name_to_index_dict[flow_seq_score_rep1_name])]
        flow_seq_score_rep2_tmp = cr_combo[int(col_name_to_index_dict[flow_seq_score_rep2_name])]
        flow_seq_score_mean_tmp = np.mean([flow_seq_score_rep1_tmp, flow_seq_score_rep2_tmp])

        cr_combo_means_tmp_dict.update({'flow_seq_score': flow_seq_score_mean_tmp})

        cr_combo_merged_list.append(cr_combo_means_tmp_dict)

    # scrreate and save final dataframe with averaged data from two replicates
    results_merged_df = pd.DataFrame(cr_combo_merged_list, index=results_df.index)
    results_merged_output_path_dir = results_df_path.replace('.csv','_final.csv')
    results_merged_df.to_csv(results_merged_output_path_dir)

    return(results_merged_df)

def create_flow_seq_score_heatmap(results_merged_df, out_path):
    '''
    create heatmaps containing flow seq scores for all CR interactions

    PARAMETERS
    --------------------
    results_merged_df: pandas dataframe
        dataframe with averaged, normalized flow seq data for all CR interactions --
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
        # sort and ignore capitalization -- by deafault, capital letters are
        # sorted first
        regulators_first_position = sorted(regulators_first_position, key=lambda v: v.upper())
        regulators_second_position = sorted(regulators_second_position, key=lambda v: v.upper())

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

    # create heatmap with flow seq score for all combinations
    plt.figure(figsize=(20,20))
    plt.rcParams.update({'font.size': 18,
                                'font.family':'helvetica'})

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

    plt.savefig(out_path+'.png', dpi=400)
    # plt.savefig(out_path+'.pdf', dpi=400)
    plt.savefig(out_path+'.svg', dpi=400)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to flow_seq_results directory')
    parser.add_argument('-r', nargs='*', help='reporter plasmids in dataset',required=True)
    parser.add_argument('-b', help='number of bins', default=8)

    args = parser.parse_args()

    results_df_path_prefix = args.i
    reporter_plasmids = args.r
    num_bins = int(args.b)

    if results_df_path_prefix[-1] != '/':
        results_df_path_prefix = results_df_path_prefix + '/'

    out_dir = results_df_path_prefix.replace('flow_seq_results', 'flow_seq_score_heatmaps')
    CHECK_FOLDER_MAIN = os.path.isdir(out_dir)
    if not CHECK_FOLDER_MAIN:
        os.makedirs(out_dir)
        print(f"created folder : {out_dir}")

    for reporter in reporter_plasmids:
        results_df_path = results_df_path_prefix + 'pMAA06_' + reporter + '_flow-seq_results.csv'
        results_merged_df = merge_replicates(results_df_path, num_bins)
        out_path = out_dir + 'pMAA06_' + reporter + '_flow-seq_scores'

        create_flow_seq_score_heatmap(results_merged_df, out_path)

if __name__ == "__main__":
    main()
