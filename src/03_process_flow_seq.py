#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on October 19 15:28:24 2022
@author: alcantar
example run 1: python 03_process_flow_seq.py -i ../../combicr_data/combicr_outputs/maa-004/pickle_files/combiCR_results.pkl -m ../../combicr_data/Quintata_NGS_raw/MAA_004/HiSeq_2022-10-26/Fastq/MAA_004_metadata.txt -r pMAA102 pMAA109 -t 100 -b 8
"""
# activate virtual enviroment before running script
# source activate CombiCR

import argparse

import pandas as pd
from matplotlib import pyplot as plt
import collections
from collections import Counter
import pickle
import os

def flow_seq_data_processing(reporter_id,
                            metadata_df,
                            barcode_count_tracker,
                            num_bins,
                            output_dir,
                            read_threshold,
                            ):

    '''
    for each flow seq replicate, this function processes normalizes read counts and calculates a
    'flow-seq score' which is essentially a weighted average of counts for each CR combination.
    normalized read counts and the flow seq score for each replicate are then merged to create
    a summary dataframe for each condition.

    PARAMETERS
    --------------------
    reporter_id: str
        plasmid used to genomically intergrate the reporter [pMAA102, pMAA109]
    metadata_df: pandas dataframe
        dataframe containing metadata
    barcode_count_tracker: dict
        dictionary containing barcode count data for all samples. this is the output
        of running 02_count_barcode.py
    num_bins: int
        number of bins used for flow seq experiment [4, 8]
    output_dir: str
        directory where flow seq results should be output
    read_threshold: int
        minimum number of reads across all bins required to keep a CR combination

    RETURNS
    --------------------
    final_df: pandas dataframe
        dataframe containing normalized counts and flow seq score for each CR combination
        and each flow-seq replicate
    '''
    # initialize a list that will contain dataframes for each flow seq replicate
    final_dfs_raw = []
    final_dfs = []

    # number of counts for each each sorting bin
    pMAA06_102_rep1_cell_counts = [800000,800000,800000,800000,
                                   800000,753794, 162269,366532]
    pMAA06_102_rep2_cell_counts = [800000,800000,800000,800000,
                                  800000,800000, 300090,361755]
    pMAA06_109_rep1_cell_counts = [800000, 800000, 800000, 800000,
                                   800000, 800000, 533757, 503151]
    pMAA06_109_rep2_cell_counts = [800000, 800000, 800000, 800000,
                                   800000, 800000, 508183, 513423]

    # fraction of cells from each sorting bin (per replicate)
    pMAA06_102_rep1_cell_fracs = [cell_count_bin/sum(pMAA06_102_rep1_cell_counts) for cell_count_bin in pMAA06_102_rep1_cell_counts]
    pMAA06_102_rep2_cell_fracs = [cell_count_bin/sum(pMAA06_102_rep2_cell_counts) for cell_count_bin in pMAA06_102_rep2_cell_counts]
    pMAA06_109_rep1_cell_fracs = [cell_count_bin/sum(pMAA06_109_rep1_cell_counts) for cell_count_bin in pMAA06_109_rep1_cell_counts]
    pMAA06_109_rep2_cell_fracs = [cell_count_bin/sum(pMAA06_109_rep2_cell_counts) for cell_count_bin in pMAA06_109_rep2_cell_counts]

    # sequencing counts for each bin
    pMAA06_102_rep1_seq_counts = [6338700, 4880776, 5620429, 5315057,
                                   6724837, 5128511, 5021194, 4821020]
    pMAA06_102_rep2_seq_counts = [4715190, 4511449, 4191818, 3909237,
                                   11304342, 11644402, 10320877, 10975460]
    pMAA06_109_rep1_seq_counts = [11964623, 9974887, 11441767, 13222435,
                                   8866989, 10537173, 10384507, 10899380]
    pMAA06_109_rep2_seq_counts = [11105437, 10966321, 10389413, 9843803,
                                    9449491, 6966687,  6284185, 6756870]
    # dictionaries with cell fraction and read count information
    cell_fracs_dict = {'pMAA102_rep_1': pMAA06_102_rep1_cell_fracs,
                        'pMAA102_rep_2': pMAA06_102_rep2_cell_fracs,
                        'pMAA109_rep_1': pMAA06_109_rep1_cell_fracs,
                        'pMAA109_rep_2': pMAA06_109_rep2_cell_fracs}
    counts_dict = {'pMAA102_rep_1': pMAA06_102_rep1_seq_counts,
                        'pMAA102_rep_2': pMAA06_102_rep2_seq_counts,
                        'pMAA109_rep_1': pMAA06_109_rep1_seq_counts,
                        'pMAA109_rep_2': pMAA06_109_rep2_seq_counts}

    # extract samples with reporter_id (pMAA102 or pMAA109) of interest and
    # create list with number of replicates (should be [1, 2])
    samples_with_reporter = metadata_df.copy()[metadata_df['genomic_reporter_id']==reporter_id]
    replicate_ids_list = list(set(list(samples_with_reporter['replicate'])))

    # process each replicate one at a time
    for replicate in replicate_ids_list:
        # extract samples corresponding to current replicate (and reporter_id)
        samples_reporter_rep = samples_with_reporter.copy()[samples_with_reporter['replicate']==replicate]
        samples_tmp = [sample.replace('_R1_001.fastq.gz','') for sample in list(samples_reporter_rep['sample_name'])]

        # create list containing all bins (should be either 4 or 8)
        bins_tmp = list(samples_reporter_rep['bin'])
        bin_names = []

        # process each bin separately
        for sample_name_tmp, bin_number in zip(samples_tmp,bins_tmp):

            CR_counts_bins = barcode_count_tracker[sample_name_tmp]['CR_combo_counts']
            bin_name = 'bin' + str(bin_number) + '_rep'+ str(replicate)
            bin_names.append(bin_name)

            # for first bin, initialize a new dataframe. for all subsequent bins, concatenate
            # new dataframe with previous dataframe
            if bin_name.split('_')[0] == 'bin1':
                flow_seq_results_master_df = pd.DataFrame.from_dict(CR_counts_bins, orient='index', columns=[bin_name])
            else:
                flow_seq_temp_df = pd.DataFrame.from_dict(CR_counts_bins, orient='index', columns=[bin_name])
                flow_seq_results_master_df = pd.concat([flow_seq_results_master_df,flow_seq_temp_df], axis=1)

        # for each row / CR combination, find sum of read counts
        while True:
            prev_shape = flow_seq_results_master_df.shape

            rows_sum = flow_seq_results_master_df.sum(axis=1, numeric_only=True)

            drop_rows = rows_sum[rows_sum <= read_threshold].index
            flow_seq_results_master_df.drop(index=drop_rows, inplace=True)

            if flow_seq_results_master_df.shape == prev_shape:
                break
        if num_bins==4: # not implemented with updated normalization methods
            # calculate flow-seq score by taking weighted average of normalized read counts
            flow_seq_results_master_df = flow_seq_results_master_df.fillna(0)
            final_dfs_raw.append(flow_seq_results_master_df)
            flow_seq_results_master_normed_df = flow_seq_results_master_df.div(flow_seq_results_master_df.sum(axis=1), axis=0)
            flow_seq_results_master_normed_df = flow_seq_results_master_normed_df.fillna(0)
            flow_seq_score_col_name = 'flow_seq_score'+'_rep_'+str(replicate)
            flow_seq_results_master_normed_df[flow_seq_score_col_name] = (flow_seq_results_master_normed_df[bin_names[0]]*0 \
                                                                       + flow_seq_results_master_normed_df[bin_names[1]]*(1/3) \
                                                                       + flow_seq_results_master_normed_df[bin_names[2]]*(2/3) \
                                                                       + flow_seq_results_master_normed_df[bin_names[3]]*(1))
        elif num_bins==8:
            normalization_id = reporter_id + '_rep_'+str(replicate)
            # calculate flow-seq score by taking weighted average of normalized read counts
            flow_seq_results_master_df = flow_seq_results_master_df.fillna(0)
            final_dfs_raw.append(flow_seq_results_master_df)
            flow_seq_results_master_normed_df = pd.DataFrame(index=flow_seq_results_master_df.index)

            for bn, bin in enumerate(flow_seq_results_master_df.columns):
                # print(bn, bin)
                # normalized each bin by number of cells and read counts
                flow_seq_results_master_normed_df[bin] = flow_seq_results_master_df[bin] * cell_fracs_dict[normalization_id][bn]/ counts_dict[normalization_id][bn]

            # divide each CR combination normalized abundance by sum of normalized
            # abundances to create fractional abundances
            flow_seq_results_master_normed_df = flow_seq_results_master_normed_df.div(flow_seq_results_master_normed_df.sum(axis=1), axis=0)
            flow_seq_results_master_normed_df = flow_seq_results_master_normed_df.fillna(0)
            flow_seq_score_col_name = 'flow_seq_score'+'_rep_'+str(replicate)
            # calculate normalized flow seq score by taking weighted average
            flow_seq_results_master_normed_df[flow_seq_score_col_name] = (flow_seq_results_master_normed_df[bin_names[0]]*0 \
                                                                       + flow_seq_results_master_normed_df[bin_names[1]]*(1/7) \
                                                                       + flow_seq_results_master_normed_df[bin_names[2]]*(2/7) \
                                                                       + flow_seq_results_master_normed_df[bin_names[3]]*(3/7) \
                                                                       + flow_seq_results_master_normed_df[bin_names[4]]*(4/7) \
                                                                       + flow_seq_results_master_normed_df[bin_names[5]]*(5/7) \
                                                                       + flow_seq_results_master_normed_df[bin_names[6]]*(6/7) \
                                                                       + flow_seq_results_master_normed_df[bin_names[7]]*(1))
        else:
            print('Flow-seq experiments should have either 4 or 8 bins. Aborting function.')
            return

        # add new dataframe to list
        final_dfs.append(flow_seq_results_master_normed_df)
    # merge both dataframes -- only keep combinations that appear in both replicates
    final_df = final_dfs[0].merge(final_dfs[1], how='inner', left_index=True, right_index=True)
    final_df_raw = final_dfs_raw[0].merge(final_dfs_raw[1], how='inner', left_index=True, right_index=True)

    export_name_norm = output_dir + 'pMAA06_' + reporter_id + '_flow-seq_results.csv'
    export_name_raw = output_dir + 'pMAA06_' + reporter_id + '_flow-seq_results_raw_counts.csv'
    variant_combos_names = final_df[((final_df['flow_seq_score_rep_1'] - final_df['flow_seq_score_rep_2']).abs() > 2/7)].index
    final_df.drop(variant_combos_names , inplace=True)
    final_df.to_csv(export_name_norm)
    final_df_raw.to_csv(export_name_raw)

    return(final_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to pickle with barcode data')
    parser.add_argument('-m', help='path to metadata')
    parser.add_argument('-r', nargs='*', help='reporter plasmids in dataset',required=True)
    parser.add_argument('-t', help='read threshold', default=100)
    parser.add_argument('-b', help='number of bins', default=8)

    args = parser.parse_args()

    barcode_count_tracker_path = args.i
    metadata_path = args.m
    reporter_plasmids = args.r
    read_threshold = int(args.t)
    num_bins = int(args.b)

    # read in metadata as pandas dataframe
    metadata_df = sample_metadata_df = pd.read_csv(metadata_path, sep='\t')

    # create output directory if it does not exist already
    flow_seq_results_out_dir = '/'.join(barcode_count_tracker_path.split('/')[0:5]) + '/flow_seq_results/'
    CHECK_FOLDER_MAIN = os.path.isdir(flow_seq_results_out_dir)
    if not CHECK_FOLDER_MAIN:
        os.makedirs(flow_seq_results_out_dir)
        print(f"created folder : {flow_seq_results_out_dir}")

    # load pickle with with count data
    with open(barcode_count_tracker_path, 'rb') as handle:
        barcode_count_tracker = pickle.load(handle)

    for reporter_id in reporter_plasmids:
        final_df = flow_seq_data_processing(reporter_id,
                                metadata_df,
                                barcode_count_tracker,
                                num_bins=num_bins,
                                output_dir=flow_seq_results_out_dir,
                                read_threshold=read_threshold)

if __name__ == "__main__":
    main()
