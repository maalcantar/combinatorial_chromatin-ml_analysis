#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 17 11:01:40 2023
@author: alcantar
example run: python protein_features_cleaning.py
"""
# activate virtual enviroment before running script
# source activate combicr

import pandas as pd
import numpy as np
import pickle
import os

def create_protein_feature_df(path_to_embeddings,
                              path_to_physiochemical_feats,
                              flow_seq_scores_df):

    '''
    create dataframes containing regulator combinatins and their corresponding embeddings and
    physiochemical properties

    PARAMETERS
    --------------------
    path_to_embeddings: string
        path to pickle file containing embeddings
    path_to_physiochemical_feats: string
        path to pickle file containing physiochemical properties for each protein
    flow_seq_scores_df: pandas dataframe
        dataframe with flow-seq scores from screen

    RETURNS
    --------------------
    final_embeddings_df: pandas dataframe
        dataframe with combinations and their corresponding feature embeddings
    '''


    # loading pickle file containing dictionary with all physiochemical features
    with open(path_to_physiochemical_feats, 'rb') as handle:
        combicr_physiochemical_features_dict = pickle.load(handle)

    # create names for physiochemical that will be used as column headers
    biochem_features_names = list(combicr_physiochemical_features_dict['sir2'].keys()) # does matter which protein is used
    biochem_features_name1 = [i + '_1' for i in biochem_features_names]
    biochem_features_name2 = [i + '_2' for i in biochem_features_names]

    # load pickle with embeddings
    with open(path_to_embeddings, 'rb') as handle:
        embeddings_raw = pickle.load(handle)

    # update empty regulator embeddings so that they are all 0
    embeddings_dimensionality = len(embeddings_raw['sir2'])
    embeddings_raw.update({'empty': [0]*embeddings_dimensionality})

    # create embeddings for all combinations
    protein_embeddings_clean_dict = dict()
    for protein_1 in embeddings_raw:
        for protein_2 in embeddings_raw:
            protein_combination_tmp = protein_1 + '_' + protein_2

            embeddings_reg1 = embeddings_raw[protein_1]
            embeddings_reg2 = embeddings_raw[protein_2]

            biochem_features_reg1 = combicr_physiochemical_features_dict[protein_1]
            biochem_features_reg1 = list(biochem_features_reg1.values())
            biochem_features_reg2 = combicr_physiochemical_features_dict[protein_2]
            biochem_features_reg2 = list(biochem_features_reg2.values())
            embeddings_concat = embeddings_reg1 + embeddings_reg2 + biochem_features_reg1 + biochem_features_reg2

            protein_embeddings_clean_dict.update({protein_combination_tmp: embeddings_concat})

    embeddings_name_1 = ['embed1_' + str(i) for i in range(1,embeddings_dimensionality+1)]
    embeddings_name_2 = [i.replace('embed1', 'embed2') for i in embeddings_name_1]

    feature_names = embeddings_name_1 + embeddings_name_2 + biochem_features_name1 + biochem_features_name2

    final_embeddings_df = pd.DataFrame.from_dict(protein_embeddings_clean_dict, orient='index',
                       columns=feature_names)

    final_embeddings_with_scores_df = final_embeddings_df.merge(flow_seq_scores_df, how='inner', left_index=True, right_index=True)

    return(final_embeddings_with_scores_df)

def main():
    # flow_seq_results_screen1 = pd.read_csv('../../../combicr_data/combicr_outputs/maa-004/flow_seq_results/pMAA06_pMAA102_flow-seq_results_final.csv', index_col=0)
    # flow_seq_results_screen1 = flow_seq_results_screen1.loc[:, ['flow_seq_score']]
    #
    # flow_seq_results_screen2 = pd.read_csv('../../../combicr_data/combicr_outputs/maa-004/flow_seq_results/pMAA06_pMAA109_flow-seq_results_final.csv', index_col=0)
    # flow_seq_results_screen2 = flow_seq_results_screen2.loc[:, ['flow_seq_score']]

    flow_seq_results_screen3 = pd.read_csv('../../../combicr_data/combicr_outputs/maa-005/flow_seq_results/pMAA06_pMAA166_flow-seq_results_final.csv', index_col=0)
    flow_seq_results_screen3 = flow_seq_results_screen3.loc[:, ['flow_seq_score']]

    flow_seq_results_screen4 = pd.read_csv('../../../combicr_data/combicr_outputs/maa-005/flow_seq_results/pMAA06_pMAA167_flow-seq_results_final.csv', index_col=0)
    flow_seq_results_screen4 = flow_seq_results_screen4.loc[:, ['flow_seq_score']]

    path_to_embeddings_esm = '../../data/esm_embeddings_raw/combicr_esm_embeddings_dict.pkl'
    path_to_embeddings_unirep64 = '../../data/unirep_embeddings_raw/unirep64/combicr_unirep_embeddings_average_hidden_64_dict.pkl'
    path_to_embeddings_unirep1900 = '../../data/unirep_embeddings_raw/unirep1900/combicr_unirep_embeddings_average_hidden_1900_dict.pkl'
    path_to_embeddings_unirep64_scrambled = '../../data/unirep_embeddings_raw/unirep64_scrambled/combicr_unirep_embeddings_scrambled_average_hidden_64_dict.pkl'

    path_to_physiochemical_feats = '../../data/physiochemical_features_raw/combicr_physiochemical_features_dict.pkl'

    esm_final_embeddings_df = create_protein_feature_df(path_to_embeddings_esm,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen1)

    unirep64_final_embeddings_df = create_protein_feature_df(path_to_embeddings_unirep64,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen1)
    unirep1900_final_embeddings_df = create_protein_feature_df(path_to_embeddings_unirep1900,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen1)

    unirep64_final_scrambled_embeddings_df = create_protein_feature_df(path_to_embeddings_unirep64_scrambled,
                                                       path_to_physiochemical_feats,
                                                      flow_seq_results_screen1)

    esm_final_embeddings_screen2_df = create_protein_feature_df(path_to_embeddings_esm,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen2)

    unirep64_final_embeddings_screen2_df = create_protein_feature_df(path_to_embeddings_unirep64,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen2)
    unirep1900_final_embeddings_screen2_df = create_protein_feature_df(path_to_embeddings_unirep1900,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen2)

    unirep64_final_scrambled_embeddings_screen2_df = create_protein_feature_df(path_to_embeddings_unirep64_scrambled,
                                                       path_to_physiochemical_feats,
                                                      flow_seq_results_screen2)

    esm_final_embeddings_screen3_df = create_protein_feature_df(path_to_embeddings_esm,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen3)

    unirep64_final_embeddings_screen3_df = create_protein_feature_df(path_to_embeddings_unirep64,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen3)
    unirep1900_final_embeddings_screen3_df = create_protein_feature_df(path_to_embeddings_unirep1900,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen3)

    unirep64_final_scrambled_embeddings_screen3_df = create_protein_feature_df(path_to_embeddings_unirep64_scrambled,
                                                       path_to_physiochemical_feats,
                                                      flow_seq_results_screen3)

    esm_final_embeddings_screen4_df = create_protein_feature_df(path_to_embeddings_esm,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen4)

    unirep64_final_embeddings_screen4_df = create_protein_feature_df(path_to_embeddings_unirep64,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen4)
    unirep1900_final_embeddings_screen4_df = create_protein_feature_df(path_to_embeddings_unirep1900,
                                                        path_to_physiochemical_feats,
                                                       flow_seq_results_screen4)

    unirep64_final_scrambled_embeddings_screen4_df = create_protein_feature_df(path_to_embeddings_unirep64_scrambled,
                                                       path_to_physiochemical_feats,
                                                      flow_seq_results_screen4)


    features_df_dir = '../../data/embedding_dataframes_clean/'
    features_df_screen2_dir = '../../data/embedding_dataframes_screen2_clean/'
    features_df_screen3_dir = '../../data/embedding_dataframes_screen3_clean/'
    features_df_screen4_dir = '../../data/embedding_dataframes_screen4_clean/'
    create ouput folders if needed
    CHECK_FOLDER_MAIN = os.path.isdir(features_df_dir)
    if not CHECK_FOLDER_MAIN:
        os.makedirs(features_df_dir)
        print(f"created folder : {features_df_dir}")
    CHECK_FOLDER_MAIN = os.path.isdir(features_df_screen2_dir)
    if not CHECK_FOLDER_MAIN:
        os.makedirs(features_df_screen2_dir)
        print(f"created folder : {features_df_screen2_dir}")
    CHECK_FOLDER_MAIN = os.path.isdir(features_df_screen3_dir)
    if not CHECK_FOLDER_MAIN:
        os.makedirs(features_df_screen3_dir)
        print(f"created folder : {features_df_screen3_dir}")
    CHECK_FOLDER_MAIN = os.path.isdir(features_df_screen4_dir)
    if not CHECK_FOLDER_MAIN:
        os.makedirs(features_df_screen4_dir)
        print(f"created folder : {features_df_screen4_dir}")

    esm_output_path = features_df_dir + 'esm_final_embeddings.csv'
    unrep64_output_path = features_df_dir + 'unirep64_final_embeddings.csv'
    unrep1900_output_path = features_df_dir + 'unirep1900_final_embeddings.csv'
    unrep64_scrambled_output_path = features_df_dir + 'unirep64_final_embeddings_scrambled.csv'

    esm_output_path_screen2 = features_df_screen2_dir + 'esm_final_screen2_embeddings.csv'
    unrep64_output_path_screen2 = features_df_screen2_dir + 'unirep64_final_screen2_embeddings.csv'
    unrep1900_output_path_screen2 = features_df_screen2_dir + 'unirep1900_final_screen2_embeddings.csv'
    unrep64_scrambled_output_path_screen2 = features_df_screen2_dir + 'unirep64_final_screen2_embeddings_scrambled.csv'

    esm_output_path_screen3 = features_df_screen3_dir + 'esm_final_screen3_embeddings.csv'
    unrep64_output_path_screen3 = features_df_screen3_dir + 'unirep64_final_screen3_embeddings.csv'
    unrep1900_output_path_screen3 = features_df_screen3_dir + 'unirep1900_final_screen3_embeddings.csv'
    unrep64_scrambled_output_path_screen3 = features_df_screen3_dir + 'unirep64_final_screen3_embeddings_scrambled.csv'

    esm_output_path_screen4 = features_df_screen4_dir + 'esm_final_screen4_embeddings.csv'
    unrep64_output_path_screen4 = features_df_screen4_dir + 'unirep64_final_screen4_embeddings.csv'
    unrep1900_output_path_screen4 = features_df_screen4_dir + 'unirep1900_final_screen4_embeddings.csv'
    unrep64_scrambled_output_path_screen4 = features_df_screen4_dir + 'unirep64_final_screen4_embeddings_scrambled.csv'

    esm_final_embeddings_df.to_csv(esm_output_path, index=True)
    unirep64_final_embeddings_df.to_csv(unrep64_output_path, index=True)
    unirep1900_final_embeddings_df.to_csv(unrep1900_output_path, index=True)
    unirep64_final_scrambled_embeddings_df.to_csv(unrep64_scrambled_output_path, index=True)

    esm_final_embeddings_df.to_csv(esm_output_path_screen2, index=True)
    unirep64_final_embeddings_screen2_df.to_csv(unrep64_output_path_screen2, index=True)
    unirep1900_final_embeddings_df.to_csv(unrep1900_output_path_screen2, index=True)
    unirep64_final_scrambled_embeddings_screen2_df.to_csv(unrep64_scrambled_output_path_screen2, index=True)

    unirep64_final_embeddings_screen3_df.to_csv(unrep64_output_path_screen3, index=True)
    unirep64_final_scrambled_embeddings_screen3_df.to_csv(unrep64_scrambled_output_path_screen3, index=True)

    unirep64_final_embeddings_screen4_df.to_csv(unrep64_output_path_screen4, index=True)
    unirep64_final_scrambled_embeddings_screen4_df.to_csv(unrep64_scrambled_output_path_screen4, index=True)

if __name__ == "__main__":
    main()
