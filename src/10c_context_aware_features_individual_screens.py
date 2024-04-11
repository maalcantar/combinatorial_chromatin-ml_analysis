#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on November 16 18:29:07 2023
@author: alcantar
example run: python 10c_context_aware_features_individual_screens.py -i ../data/machine_learning_context_aware/random_forest_regression/random_forest_regression_model.pkl -n 100 --save 1
"""
# activate virtual enviroment before running script
# source activate combicr_ml

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from sklearn.metrics import r2_score

import os
import pickle
import argparse


def add_screen_context_feature(screen_results_path, screen_no):

    '''
    creates a pandas dataframe containing flow seq scores, protein embeddings,
    and an additional one-hot encoded feature specifying the screening context

    PARAMETERS
    --------------------
    flow_seq_results_df: pandas dataframe
        dataframe with merged flow seq results and features (i.e., embeddings)
    screen_no: int
        screen number

    RETURNS
    --------------------
    screen_results_df: pandas dataframe
        flow seq results with embeddings and additional feature specifying
        the screen context

    '''

    screen_results_df = pd.read_csv(screen_results_path, index_col=0)
    no_embeddings = 128

    screen_results_df.index = screen_results_df.index + f'_{str(screen_no)}'
    for screen_no_temp in range(1,5):
        insert_idx = no_embeddings + screen_no_temp - 1
        screen_name = 'screen_' + str(screen_no_temp)
        if screen_no_temp == screen_no:
            screen_results_df.insert(loc=insert_idx, column=screen_name, value=1)
        else:
            screen_results_df.insert(loc=insert_idx, column=screen_name, value=0)

    return(screen_results_df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to random forest model')
    parser.add_argument('-n', help='number of trials')
    parser.add_argument('--save', help='save importances?', default = 0)

    args = parser.parse_args()
    RF_model_path = args.i
    ntrials =  int(args.n)#100 # int(args.n)
    save_importances = int(args.save) #True #int(args.save)

    outname = 'feature_importances_context_aware'
    outdir = RF_model_path.replace('random_forest_regression_model.pkl','')

    # load in the model
    with open(RF_model_path, 'rb') as handle:
            RF_model = pickle.load(handle)

    screen1_results_path = '../data/embedding_dataframes_clean/unirep64_final_embeddings.csv'
    screen2_results_path = '../data/embedding_dataframes_screen2_clean/unirep64_final_screen2_embeddings.csv'
    screen3_results_path = '../data/embedding_dataframes_screen3_clean/unirep64_final_screen3_embeddings.csv'
    screen4_results_path = '../data/embedding_dataframes_screen4_clean/unirep64_final_screen4_embeddings.csv'

    screen_results_paths_list = [screen1_results_path, screen2_results_path,
                                screen3_results_path, screen4_results_path]

    screen_results_list = []
    for screen_number, screen_results_path in enumerate(screen_results_paths_list, 1):
        screen_results_df_tmp = add_screen_context_feature(screen_results_path=screen_results_path,
                                                     screen_no=screen_number)

        screen_results_list.append(screen_results_df_tmp)

    screen_results_all = pd.concat(screen_results_list)
    combo_only_list = []
    combo_all_overlap = []
    for combo_with_screen in screen_results_all.index:
        combo_only = '_'.join(combo_with_screen.split('_')[0:2])
        combo_only_list.append(combo_only)

    for x in set(combo_only_list):
        if combo_only_list.count(x) > 3:
            combo_all_overlap.append(x)

    screen_results_all_overlap_df = screen_results_all.copy()[screen_results_all.index.str.contains('|'.join(combo_all_overlap))]

    for screen_no in range(1,5):
        print(f'starting screen {str(screen_no)}')
        screen_no_suffix = '_' + str(screen_no)

        screen_results_all_subset_df = screen_results_all_overlap_df.copy()[screen_results_all_overlap_df.index.str.contains(screen_no_suffix)]

        # split training features and target values
        unirep64_feats = 132 # excluding physiochemical features
        X_df = screen_results_all_subset_df.iloc[:,0:unirep64_feats].dropna(axis=1)#
        y_df = screen_results_all_subset_df.copy().iloc[:, [-1]]

        # create train-test-splits
        test_size = 0.20
        rf_feature_importance_df = pd.DataFrame()
        r2_RF_df = pd.DataFrame()
        r2_RF_list = []
        for trial in range(ntrials):
            if trial % 5 == 0:
                print(trial)
            random_state = trial
            trial_name = "trial_" + str(trial)
            X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                                test_size=test_size,
                                                               random_state=random_state)

            RF_model.fit(X_train, y_train.values.ravel())
            y_test_RF_pred = RF_model.predict(X_test)
            r2_RF = round(r2_score(list(y_test['flow_seq_score']), y_test_RF_pred),3)
            r2_RF_list.append(r2_RF)
            feature_importances_tmp = RF_model.feature_importances_
            rf_feature_importance_df[trial_name] = feature_importances_tmp

        # rename indices with their corresponding embedding names
        r2_RF_df['R2_values'] = r2_RF_list
        embed_names_list_part1 =  ['embed1_' + str(i) for i in range(1,65)]
        embed_names_list_part2 =  ['embed2_' + str(i) for i in range(1,65)]
        embed_names_list_part3 =  ['context_' + str(i) for i in range(1,5)]
        embed_names_list = embed_names_list_part1 + embed_names_list_part2 + embed_names_list_part3

        # sum feature importances across trials
        rf_feature_importance_df = rf_feature_importance_df.rename(index=dict(zip(range(0,132), embed_names_list)))
        rf_feature_importance_df = pd.DataFrame(rf_feature_importance_df.sum(axis=1), columns=['feature_importance'])

        # save reults to csv
        ntrials_path_name = '_ntrials' + str(ntrials) + 'screen' + screen_no_suffix
        outname_full = outdir + outname + ntrials_path_name

        if save_importances:
            print('saving feature importances')
            rf_feature_importance_df.to_csv(outname_full + '.csv')
            r2_RF_df.to_csv(outname_full+ '_R2_scores.csv')

if __name__ == "__main__":
    main()
