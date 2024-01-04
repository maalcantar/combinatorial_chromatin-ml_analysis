#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on February 14 14:14:20 2023
@author: alcantar
example run: python 09_predict_missing_data.py
"""
# activate virtual enviroment before running script
# source activate combicr_ml


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from scipy import stats
from sklearn.metrics import r2_score
import seaborn as sns

import os
import pickle
import argparse

def main():
    print('loading models')
    RF_model_path = '../data/machine_learning/random_forest_regression/random_forest_regression_model.pkl'
    with open(RF_model_path, 'rb') as handle:
            RF_model = pickle.load(handle)

    print('loading data')
    embeddings_path = '../data/embedding_dataframes_clean/unirep64_final_embeddings.csv'
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)
    embeddings_df = embeddings_df.iloc[:, np.r_[:128, -1]]
    unirep64_feats = 128


    print('creating testing data')
    all_combos_embeddings_path =  '../data/unirep_embeddings_raw/unirep64/combicr_unirep_embeddings_average_hidden_64_dict.pkl'
    with open(all_combos_embeddings_path, 'rb') as handle:
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

            embeddings_concat = embeddings_reg1 + embeddings_reg2

            protein_embeddings_clean_dict.update({protein_combination_tmp: embeddings_concat})

    embeddings_name_1 = ['embed1_' + str(i) for i in range(1,embeddings_dimensionality+1)]
    embeddings_name_2 = [i.replace('embed1', 'embed2') for i in embeddings_name_1]

    feature_names = embeddings_name_1 + embeddings_name_2

    final_embeddings_all_df = pd.DataFrame.from_dict(protein_embeddings_clean_dict, orient='index',
                           columns=feature_names)

    combos_in_screen = list(embeddings_df.index)
    combos_missing_from_screen_df = final_embeddings_all_df.copy().drop(combos_in_screen)
    combos_missing_from_screen_df = combos_missing_from_screen_df.loc[(~combos_missing_from_screen_df.index.str.contains('cdc36_')) & (~combos_missing_from_screen_df.index.str.contains('_hda1'))]

    combos_missing_from_screen_df = combos_missing_from_screen_df.iloc[:, np.r_[:128, -1]]
    unirep64_feats = 128

    print('creating training data')
    X_test = combos_missing_from_screen_df.iloc[:,0:unirep64_feats].dropna(axis=1)
    embeddings_df = embeddings_df.sample(frac=1) # 0.8 gives nearly identical results

    print('creating training data')
    X_train = embeddings_df.iloc[:,0:unirep64_feats].dropna(axis=1)
    y_train = embeddings_df.copy().iloc[:, [-1]]

    RF_model.fit(X_train, y_train.values.ravel())

    missing_data_preds = RF_model.predict(X_test)

    missing_data_preditions_df = pd.DataFrame(index = X_test.index)
    missing_data_preditions_df['y_pred'] = missing_data_preds

    missing_data_preditions_df.to_csv('../missing_data_predictions.csv')

if __name__ == "__main__":
    main()
