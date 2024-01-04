#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on October 20 11:23:31 2022
@author: alcantar
example run 1: python 04_generate_flowseq_qc_plots.py -i ../../combicr_data/combicr_outputs/maa-004/flow_seq_results/ -r pMAA102 pMAA109 -b 8
"""
# activate virtual enviroment before running script
# source activate CombiCR

import argparse

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os

import seaborn as sns

from scipy.stats import pearsonr
from scipy.stats import spearmanr


def plot_correlations(plasmid_reporters, num_bins, flow_seq_results_dir, qc_out_dir):
    '''
    plot all relevant correlations between replicates: flow seq score correlation
    and normalized counts per bin

    PARAMETERS
    --------------------
    plasmid_reporters: list of strs
        plasmid reporters to analyze [pMAA102, pMAA109]
    num_bins: int
        number of bins used for flow seq experiment [4, 8]
    flow_seq_results_dir: str
        path to directory with flow seq results csv files
    qc_out_dir: str
        directory where flow seq correlations will be written

    RETURNS
    --------------------
    NONE: just creates plots

    '''


    # analyze each reporter architecture individually
    for plasmid_id in plasmid_reporters:

        # subfolder for each reporter
        plasmid_qc_out_dir = qc_out_dir + plasmid_id + '/'
        CHECK_FOLDER_MAIN = os.path.isdir(plasmid_qc_out_dir)
        if not CHECK_FOLDER_MAIN:
            os.makedirs(plasmid_qc_out_dir)
            print(f"created folder : {plasmid_qc_out_dir}")

        # read in normalized flow seq dataframe
        flow_seq_results_file_name = 'pMAA06_' + plasmid_id + '_flow-seq_results.csv'
        flow_seq_results_file_path = flow_seq_results_dir + flow_seq_results_file_name
        flow_seq_results_df = pd.read_csv(flow_seq_results_file_path, index_col=0)

        if num_bins==4:
            fig, axes = plt.subplots(2, 2, figsize=(16,8))
        elif num_bins==8:
            fig, axes = plt.subplots(4, 2,  figsize=(16,8))

        plt.figure(figsize=(10,8))
        plt.rcParams.update({'font.size': 18,
                         'font.family':'arial'})
        counter_row=0
        counter_col=0
        # create correlation plot for each bin
        for bin_number in range(1, num_bins+1):
            rep1_bin = 'bin' + str(bin_number) + '_rep1'
            rep2_bin = 'bin' + str(bin_number) + '_rep2'
            scatter = sns.scatterplot(ax=axes[counter_row,counter_col], data=flow_seq_results_df, x=rep1_bin, y=rep2_bin, color='k', s=40)
            scatter.set(ylim=(0, 1))
            scatter.set(xlim=(0, 1))
            scatter.set_title(rep1_bin[0:4])
            pearson_r_val, pearson_p_val = pearsonr(flow_seq_results_df[rep1_bin], flow_seq_results_df[rep2_bin])
            spearman_r_val, speaerman_p_val = spearmanr(flow_seq_results_df[rep1_bin], flow_seq_results_df[rep2_bin])
            scatter.text(0.1, 0.8, f'R$^2$: {str(round(pearson_r_val**2,3))} ', fontsize=18)
            scatter.text(0.1, 0.7, f'Spearman $\\rho$: {str(round(spearman_r_val,3))} ', fontsize=18) #add text

            if counter_col >=1:
                counter_col = 0
                counter_row+=1
            else:
                counter_col+=1

            out_fig_name_bins = plasmid_qc_out_dir + 'normalized_bin_counts_corr'

        fig.tight_layout()
        fig.savefig(out_fig_name_bins + '.png', dpi=400)
        fig.savefig(out_fig_name_bins + '.pdf', dpi=400)
        fig.savefig(out_fig_name_bins + '.svg', dpi=400)
        plt.close('all')

        plt.figure(figsize=(10,8))
        plt.rcParams.update({'font.size': 18,
                         'font.family':'arial'})

        # create correlation plot for each flow seq score
        flow_set_score_rep1 = 'flow_seq_score' + '_rep_1'
        flow_set_score_rep2 = 'flow_seq_score' + '_rep_2'
        scatter = sns.scatterplot(data=flow_seq_results_df, x=flow_set_score_rep1, y=flow_set_score_rep2, color='k', s=40)
        scatter.set(ylim=(0, 1))
        scatter.set(xlim=(0, 1))
        scatter.set_title('flow-seq score correlation')
        pearson_r_val, pearson_p_val = pearsonr(flow_seq_results_df[flow_set_score_rep1], flow_seq_results_df[flow_set_score_rep2])
        spearman_r_val, speaerman_p_val = spearmanr(flow_seq_results_df[flow_set_score_rep1], flow_seq_results_df[flow_set_score_rep2])
        num_CRs_assayed = 48
        num_potential_combos = num_CRs_assayed**2
        num_combos = flow_seq_results_df.shape[0]
        percent_combos = num_combos / num_potential_combos * 100
        scatter.text(0.07, 0.95, f'R$^2$: {str(round(pearson_r_val**2,3))} ', fontsize=18)
        scatter.text(0.07, 0.88, f'Spearman $\\rho$: {str(round(spearman_r_val,3))} ', fontsize=18)
        scatter.text(0.07, 0.81, f'Configurations: {str(num_combos)} ({str(round(percent_combos,1))}%)', fontsize=18)
        out_fig_name_flow_score = plasmid_qc_out_dir + 'flow_seq_score_corr'

        plt.tight_layout()
        plt.savefig(out_fig_name_flow_score + '.png', dpi=400)
        plt.savefig(out_fig_name_flow_score + '.pdf', dpi=400)
        plt.savefig(out_fig_name_flow_score + '.svg', dpi=400)
        plt.close('all')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to flow_seq_results_directory')
    parser.add_argument('-r', nargs='*', help='reporter plasmids in dataset',required=True)
    parser.add_argument('-b', help='number of bins', default=8)

    args = parser.parse_args()

    flow_seq_results_dir = args.i
    plasmid_reporters = args.r
    num_bins = int(args.b)

    if flow_seq_results_dir[-1] != '/':
        flow_seq_results_dir = flow_seq_results_dir + '/'

    qc_out_dir = flow_seq_results_dir.replace('flow_seq_results', 'qc_plots')
    CHECK_FOLDER_MAIN = os.path.isdir(qc_out_dir)
    if not CHECK_FOLDER_MAIN:
        os.makedirs(qc_out_dir)
        print(f"created folder : {qc_out_dir}")

    plot_correlations(plasmid_reporters, num_bins,
    flow_seq_results_dir, qc_out_dir)

if __name__ == "__main__":
    main()
