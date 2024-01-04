#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on November 1 15:24:50 2022
@author: alcantar
example run: python 06_calc_bliss_scores.py -i ../../combicr_data/combicr_outputs/maa-004/flow_seq_results/ -r pMAA102 pMAA109
"""
# activate virtual enviroment before running script
# source activate CombiCR

import argparse

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os

def bliss_score_calc_plot(flow_seq_results_df, out_dir_dfs,
                          out_dir_plots, reporter):

    '''
    calculates bliss scores for all possible combinations (i.e., those where both
    regulators in a combination were also paired with an empty zinc finger)

    PARAMETERS
    --------------------
    flow_seq_results_df: pandas dataframe
        dataframe with all flow seq scores
    out_dir_dfs: str
        output directory for dataframes
    out_dir_plots: str
        output directory for plots
    reporter: str
        current reporter name

    RETURNS
    --------------------
    None - just saves dataframe and plots
    '''

    # extract single regulator activities
    flow_seq_pos_1_scores_df = flow_seq_results_df.copy()[flow_seq_results_df.index.str.contains('_empty')]
    flow_seq_pos_2_scores_df = flow_seq_results_df.copy()[flow_seq_results_df.index.str.contains('empty_')]
    pos1_regs = [CR.split('_')[0] for CR in flow_seq_pos_1_scores_df.index]
    pos2_regs = [CR.split('_')[1] for CR in flow_seq_pos_2_scores_df.index]
    pos_1_scores = dict(zip(pos1_regs, flow_seq_pos_1_scores_df['flow_seq_score']))
    pos_2_scores = dict(zip(pos2_regs, flow_seq_pos_2_scores_df['flow_seq_score']))
    bliss_scores_dict = dict()

    # loop through all combinations and calculate bliss sum scores
    for CR_combo in flow_seq_results_df.itertuples():
        # extract combination information
        CR_combo_id = CR_combo[0].split('_')
        CR_combo_pos1 = CR_combo_id[0]
        CR_combo_pos2 = CR_combo_id[1]
        CR_combo_score = CR_combo[9]

        try:
            CR_1_alone_score = pos_1_scores[CR_combo_pos1]
            CR_2_alone_score = pos_2_scores[CR_combo_pos2]

            # bliss score calculation
            bliss_score_pred = CR_1_alone_score + CR_2_alone_score - CR_1_alone_score*CR_2_alone_score
            bliss_score_tmp = CR_combo_score - bliss_score_pred
            # bliss_score_tmp = ((CR_combo_score-1) - (CR_1_alone_score-1) - (CR_2_alone_score-1))/(CR_combo_score-1)
            bliss_scores_dict.update({CR_combo[0]: bliss_score_tmp})
        except KeyError:
            continue

    bliss_scores_df = pd.DataFrame(bliss_scores_dict.items())
    bliss_scores_df = bliss_scores_df.rename(index=dict(enumerate(bliss_scores_df[0]))).drop([0], axis=1).rename(columns={1:'bliss_scores'})

    # plot bliss scores
    plt.rcParams.update({'font.size': 18,
                         'font.family':'sans-serif'})

    fig, axes = plt.subplots(2, 1,  gridspec_kw={"height_ratios":(.30, .10)}, figsize = (13, 10))

    histogram = sns.histplot(x=bliss_scores_df['bliss_scores'], color='cornflowerblue', ax=axes[0])
    boxplot = sns.boxplot(x=bliss_scores_df['bliss_scores'], color='cornflowerblue', ax=axes[1])

    histogram.set_xlim(-1, 0.8)
    histogram.set_ylim(0, 275)
    boxplot.set_xlim(-1, 0.8)

    plt.savefig(out_dir_plots + reporter + '_bliss_score.pdf')
    plt.savefig(out_dir_plots + reporter + '_bliss_score.png')
    plt.savefig(out_dir_plots + reporter + '_bliss_score.svg')

    bliss_scores_df.to_csv(out_dir_dfs + reporter + '_bliss_score.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to flow_seq_results directory')
    parser.add_argument('-r', nargs='*', help='reporter plasmids in dataset',required=True)

    args = parser.parse_args()

    flow_seq_results_dir = args.i
    reporter_id = args.r

    if flow_seq_results_dir[-1] != '/':
        flow_seq_results_dir = flow_seq_results_dir + '/'

    # create output directories for dataframes and plots
    out_dir_dfs = flow_seq_results_dir.replace('flow_seq_results', 'bliss_scores_dfs')
    CHECK_FOLDER_MAIN_DF = os.path.isdir(out_dir_dfs)
    if not CHECK_FOLDER_MAIN_DF:
        os.makedirs(out_dir_dfs)
        print(f"created folder : {out_dir_dfs}")

    out_dir_plots = flow_seq_results_dir.replace('flow_seq_results', 'bliss_scores_plots')
    CHECK_FOLDER_MAIN_PLOT = os.path.isdir(out_dir_plots)
    if not CHECK_FOLDER_MAIN_PLOT:
        os.makedirs(out_dir_plots)
        print(f"created folder : {out_dir_plots}")

    # calculate bliss scores for all screens
    for reporter in reporter_id:
        flow_seq_results_path = flow_seq_results_dir + 'pMAA06_' + reporter + '_flow-seq_results_final.csv'
        flow_seq_results_df = pd.read_csv(flow_seq_results_path, index_col=0)

        bliss_score_calc_plot(flow_seq_results_df, out_dir_dfs,
                              out_dir_plots, reporter)

if __name__ == "__main__":
    main()
