#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on November 9 21:25:40 2023
@author: alcantar
example run: python 07d_rf_feature_importance.py -i ../data/embedding_dataframes_clean/unirep64_final_embeddings.csv -n 100 -s 1 --save 1
"""
# activate virtual enviroment before running script
# source activate combicr_ml

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

import os
import pickle
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to embeddings dataframe')
    parser.add_argument('-n', help='number of trials')
    parser.add_argument('-s', help='screen number')
    parser.add_argument('--save', help='save importances?', default = 0)

    args = parser.parse_args()
    embeddings_path = args.i
    embeddings_scrambled_path = args.s
    ntrials = int(args.n)
    screen_number = int(args.s)
    save_importances = int(args.save)

    # use the screen number to acquire the path to the model and to specify the
    # output directory
    if screen_number==1:
        outdir = '../data/machine_learning/random_forest_regression/'
        RF_model_path = '../data/machine_learning/random_forest_regression/random_forest_regression_model.pkl'
    elif screen_number==2:
        outdir = '../data/machine_learning_screen2/random_forest_regression/'
        RF_model_path = '../data/machine_learning_screen2/random_forest_regression/random_forest_regression_model.pkl'
    elif screen_number==3:
        outdir = '../data/machine_learning_screen3/random_forest_regression/'
        RF_model_path = '../data/machine_learning_screen3/random_forest_regression/random_forest_regression_model.pkl'
    elif screen_number==4:
        outdir = '../data/machine_learning_screen4/random_forest_regression/'
        RF_model_path = '../data/machine_learning_screen4/random_forest_regression/random_forest_regression_model.pkl'


    outname = 'feature_importances'

    # load in the model
    with open(RF_model_path, 'rb') as handle:
            RF_model = pickle.load(handle)

    # load in embeddings dataframe
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)

    # split training features and target values
    unirep64_feats = 128 # excluding physiochemical features
    X_df = embeddings_df.iloc[:,0:unirep64_feats].dropna(axis=1)#
    y_df = embeddings_df.copy().iloc[:, [-1]]

    # create train-test-splits
    test_size = 0.20

    # run n number of trials
    rf_feature_importance_df = pd.DataFrame()
    for trial in range(ntrials):
        if trial % 5 == 0:
            print(trial)
        random_state = trial
        trial_name = "trial_" + str(trial)
        X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                            test_size=test_size,
                                                           random_state=random_state)
        RF_model.fit(X_train, y_train.values.ravel())
        feature_importances_tmp = RF_model.feature_importances_
        rf_feature_importance_df[trial_name] = feature_importances_tmp

    # rename indices with their corresponding embedding names
    embed_names_list_part1 =  ['embed1_' + str(i) for i in range(1,65)]
    embed_names_list_part2 =  ['embed2_' + str(i) for i in range(1,65)]
    embed_names_list = embed_names_list_part1 + embed_names_list_part2

    # sum feature importances across trials
    rf_feature_importance_df = rf_feature_importance_df.rename(index=dict(zip(range(0,128), embed_names_list)))
    rf_feature_importance_df = pd.DataFrame(rf_feature_importance_df.sum(axis=1), columns=['feature_importance'])

    # save reults to csv
    ntrials_path_name = '_ntrials' + str(ntrials)
    outname_full = outdir + outname + ntrials_path_name

    if save_importances:
        print('saving feature importances')
        rf_feature_importance_df.to_csv(outname_full + '.csv')

if __name__ == "__main__":
    main()
