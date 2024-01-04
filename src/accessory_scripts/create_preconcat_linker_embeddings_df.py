#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on February 24 10:23:01 2023
@author: alcantar
example run: python create_preconcat_linker_embeddings_df.py
"""
# activate virtual enviroment before running script
# source activate combicr_ml

import pandas as pd
import pickle

def main():

    # create paths to non-scrambled unirep embeddings for all CR combinations
    path_to_unirep_embed_part1 = '../../data/unirep_embeddings_raw/unirep64/combicr_unirep_preconcat_linker_embeddings_average_hidden_64_dict_part1.pkl'
    path_to_unirep_embed_part2 = '../../data/unirep_embeddings_raw/unirep64/combicr_unirep_preconcat_linker_embeddings_average_hidden_64_dict_part2.pkl'

    # create paths to scrambled unirep embeddings for all CR combinations
    path_to_scrambled_unirep_embed_part1 = '../../data/unirep_embeddings_raw/unirep64/combicr_unirep_preconcat_linker_scrambled_embeddings_average_hidden_64_dict_part1.pkl'
    path_to_scrambled_unirep_embed_part2 = '../../data/unirep_embeddings_raw/unirep64/combicr_unirep_preconcat_linker_scrambled_embeddings_average_hidden_64_dict_part2.pkl'

    # load non-scrambled data from pickle
    with open(path_to_unirep_embed_part1,'rb') as handle:
        unirep_embed_part1_dict = pickle.load(handle)
    with open(path_to_unirep_embed_part2,'rb') as handle:
        unirep_embed_part2_dict = pickle.load(handle)

    # scrambled data from pickle
    with open(path_to_scrambled_unirep_embed_part1,'rb') as handle:
        unirep_scrambled_embed_part1_dict = pickle.load(handle)
    with open(path_to_scrambled_unirep_embed_part2,'rb') as handle:
        unirep_scrambled_embed_part2_dict = pickle.load(handle)

    # create complete non-scrabled dictionary
    unirep_embed_dict = dict()
    unirep_embed_dict.update(unirep_embed_part1_dict)
    unirep_embed_dict.update(unirep_embed_part2_dict)

    # create complete scrabled dictionary
    unirep_scrambled_embed_dict = dict()
    unirep_scrambled_embed_dict.update(unirep_scrambled_embed_part1_dict)
    unirep_scrambled_embed_dict.update(unirep_scrambled_embed_part2_dict)

    # create embedding names
    embeddings_dimensionality = 64
    embeddings_name = ['embed1_' + str(i) for i in range(1,embeddings_dimensionality+1)]

    flow_seq_results_screen1 = pd.read_csv('../../../combicr_data/combicr_outputs/maa-004/flow_seq_results/pMAA06_pMAA102_flow-seq_results_final.csv', index_col=0)
    flow_seq_results_screen1 = flow_seq_results_screen1.loc[:, ['flow_seq_score']]

    flow_seq_results_screen2 = pd.read_csv('../../../combicr_data/combicr_outputs/maa-004/flow_seq_results/pMAA06_pMAA109_flow-seq_results_final.csv', index_col=0)
    flow_seq_results_screen2 = flow_seq_results_screen2.loc[:, ['flow_seq_score']]


    # create dataframe with all embeddings
    final_embeddings_df = pd.DataFrame.from_dict(unirep_embed_dict, orient='index',
                           columns=embeddings_name)
    final_scrambled_embeddings_df = pd.DataFrame.from_dict(unirep_scrambled_embed_dict, orient='index',
                           columns=embeddings_name)

    # merge data with flow-seq results from screen 1
    final_embeddings_with_scores_screen1_df = final_embeddings_df.merge(flow_seq_results_screen1, how='inner', left_index=True, right_index=True)
    final_scrambled_embeddings_with_scores_screen1_df = final_scrambled_embeddings_df.merge(flow_seq_results_screen1, how='inner', left_index=True, right_index=True)

    # merge data with flow-seq results from screen 2
    final_embeddings_with_scores_screen2_df = final_embeddings_df.merge(flow_seq_results_screen2, how='inner', left_index=True, right_index=True)
    final_scrambled_embeddings_with_scores_screen2_df = final_scrambled_embeddings_df.merge(flow_seq_results_screen2, how='inner', left_index=True, right_index=True)

    # screen 1 and 2 output paths
    features_df_dir = '../../data/embedding_dataframes_clean/'
    features_df_screen2_dir = '../../data/embedding_dataframes_screen2_clean/'
    unrep64_output_screen1_path = features_df_dir + 'unirep64_preconcat_linker_final_embeddings.csv'
    unrep64_output_scrambled_screen1_path = features_df_dir + 'unirep64_preconcat_linker_scrambled_final_embeddings.csv'
    unrep64_output_screen2_path = features_df_screen2_dir + 'unirep64_preconcat_linker_final_screen2_embeddings.csv'
    unrep64_output_scrambled_screen2_path = features_df_screen2_dir + 'unirep64_preconcat_linker_scrambled_final_screen2_embeddings.csv'

    # export screen 1 data to csv
    final_embeddings_with_scores_screen1_df.to_csv(unrep64_output_screen1_path, index=True)
    final_scrambled_embeddings_with_scores_screen1_df.to_csv(unrep64_output_scrambled_screen1_path, index=True)

    # export screen 2 data to csv
    final_embeddings_with_scores_screen2_df.to_csv(unrep64_output_screen2_path, index=True)
    final_scrambled_embeddings_with_scores_screen2_df.to_csv(unrep64_output_scrambled_screen2_path, index=True)

if __name__ == "__main__":
    main()
