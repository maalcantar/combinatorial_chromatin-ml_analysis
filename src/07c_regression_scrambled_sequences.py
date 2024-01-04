#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 19 11:55:04 2023
@author: alcantar
example run: python 07c_regression_scrambled_sequences.py -i ../data/embedding_dataframes_clean/unirep64_final_embeddings.csv -s ../data/embedding_dataframes_clean/unirep64_final_embeddings_scrambled.csv -n 100 --screen 1
example run 2: python 07c_regression_scrambled_sequences.py -i ../data/embedding_dataframes_screen2_clean/unirep64_final_screen2_embeddings.csv -s ../data/embedding_dataframes_screen2_clean/unirep64_final_screen3_embeddings_scrambled.csv -n 100 --screen 2
example run 3: python 07c_regression_scrambled_sequences.py -i ../data/embedding_dataframes_screen3_clean/unirep64_final_screen3_embeddings.csv -s ../data/embedding_dataframes_screen3_clean/unirep64_final_screen3_embeddings_scrambled.csv -n 100 --screen 3
example run 4: python 07c_regression_scrambled_sequences.py -i ../data/embedding_dataframes_screen4_clean/unirep64_final_screen4_embeddings.csv -s ../data/embedding_dataframes_screen4_clean/unirep64_final_screen4_embeddings_scrambled.csv -n 100 --screen 3
"""
# activate virtual enviroment before running script
# source activate combicr_ml

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import sklearn
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

from sklearn.metrics import r2_score

def scramble_predictions(X_train, X_test, y_train, y_test,
                        X_scrambled_train, X_scrambled_test,
                        y_scrambled_train, y_scrambled_test,
                        lasso_model, RF_model,NN_model, DT_model, SVM_model):

    '''
    compare model performance between scrambled and non-scrambled amino acid sequences

    PARAMETERS
    --------------------
    X_train: pandas dataframe
        training features
    X_test: pandas dataframe
        testing features
    y_train: pandas dataframe
        training target values
    y_test: pandas dataframe
        testing target values
    X_scrambled_train: pandas dataframe
        training features for scrambled sequences
    X_scrambled_test: pandas dataframe
        testing features for scrambled sequences
    y_crambled_train: pandas dataframe
        training target values for scrambled sequences
    y_crambled_test: pandas dataframe
        testing target values for scrambled sequences
    lasso_model: sklearn model
        linear regression model
    RF_model: sklearn model
        random forest regression model
    NN_model: sklearn model
        nearest neighbors regression model
    DT_model: sklearn model
        decision tree regression model
    SVM_model: sklearn model
        support vector regression model


    RETURNS
    --------------------
    r2_lasso: list of floats
        performance for linear regression model
    r2_RF: list of floats
        performance for random forest regression model
    r2_NN: list of floats
        performance for nearest neighbors regression model
    r2_DT: list of floats
        performance for decision tree regression model
    r2_SVM: list of floats
        performance for support vector regression model
    r2_lasso_scram: list of floats
        performance for linear regression model with scrambled sequences
    r2_RF_scram: list of floats
        performance for random forest regression model with scrambled sequences
    r2_NN_scram: list of floats
        performance for nearest neighbors regression model with scrambled sequences
    r2_DT_scram: list of floats
        performance for decision tree regression model with scrambled sequences
    r2_SVM_scram: list of floats
        performance for support vector regression model with scrambled sequences

    '''

    lasso_model.fit(X_train, y_train.values.ravel())
    y_test_lasso_pred = lasso_model.predict(X_test)

    RF_model.fit(X_train, y_train.values.ravel())
    y_test_RF_pred = RF_model.predict(X_test)

    NN_model.fit(X_train, y_train.values.ravel())
    y_test_NN_pred = NN_model.predict(X_test)

    DT_model.fit(X_train, y_train.values.ravel())
    y_test_DT_pred = DT_model.predict(X_test)

    SVM_model.fit(X_train, y_train.values.ravel())
    y_test_SVM_pred = SVM_model.predict(X_test)

    r2_lasso = round(r2_score(list(y_test['flow_seq_score']), y_test_lasso_pred),3)
    r2_RF = round(r2_score(list(y_test['flow_seq_score']), y_test_RF_pred),3)
    r2_NN = round(r2_score(list(y_test['flow_seq_score']), y_test_NN_pred),3)
    r2_DT = round(r2_score(list(y_test['flow_seq_score']), y_test_DT_pred),3)
    r2_SVM = round(r2_score(list(y_test['flow_seq_score']), y_test_SVM_pred),3)

    lasso_model.fit(X_scrambled_train, y_scrambled_train.values.ravel())
    y_test_lasso_pred_scram = lasso_model.predict(X_scrambled_test)

    RF_model.fit(X_scrambled_train, y_scrambled_train.values.ravel())
    y_test_RF_pred_scram = RF_model.predict(X_scrambled_test)

    NN_model.fit(X_scrambled_train, y_scrambled_train.values.ravel())
    y_test_NN_pred_scram = NN_model.predict(X_scrambled_test)

    DT_model.fit(X_scrambled_train, y_scrambled_train.values.ravel())
    y_test_DT_pred_scram = DT_model.predict(X_scrambled_test)

    SVM_model.fit(X_scrambled_train, y_scrambled_train.values.ravel())
    y_test_SVM_pred_scram = SVM_model.predict(X_scrambled_test)

    r2_lasso_scram = round(r2_score(list(y_scrambled_test['flow_seq_score']), y_test_lasso_pred_scram),3)
    r2_RF_scram = round(r2_score(list(y_scrambled_test['flow_seq_score']), y_test_RF_pred_scram),3)
    r2_NN_scram = round(r2_score(list(y_scrambled_test['flow_seq_score']), y_test_NN_pred_scram),3)
    r2_DT_scram = round(r2_score(list(y_scrambled_test['flow_seq_score']), y_test_DT_pred_scram),3)
    r2_SVM_scram = round(r2_score(list(y_scrambled_test['flow_seq_score']), y_test_SVM_pred_scram),3)

    return(r2_lasso, r2_RF, r2_NN, r2_DT,r2_SVM, r2_lasso_scram, r2_RF_scram, r2_NN_scram, r2_DT_scram, r2_SVM_scram)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to embeddings dataframe')
    parser.add_argument('-s', help='path to scrambled embeddings dataframe')
    parser.add_argument('-n', help='number of trials')
    parser.add_argument('--screen', help='screen number', required=True)

    args = parser.parse_args()
    embeddings_path = args.i
    embeddings_scrambled_path = args.s
    ntrials = int(args.n)
    screen_number = int(args.screen)

    embeddings_df = pd.read_csv(embeddings_path, index_col=0)
    embeddings_scrambed_df = pd.read_csv(embeddings_scrambled_path, index_col=0)

    lasso_model_path = '../data/machine_learning/lasso_regression/lasso_regression_model.pkl'
    RF_model_path = '../data/machine_learning/random_forest_regression/random_forest_regression_model.pkl'
    NN_model_path = '../data/machine_learning/nearest_neighbors_regression/KNN_regression_model.pkl'
    DT_model_path = '../data/machine_learning/decision_tree_regression/decision_tree_regression_model.pkl'
    SVM_model_path = '../data/machine_learning/support_vector_regression/support_vector_regression_model.pkl'

    if screen_number !=1:
        name_suffix = 'machine_learning_screen' + str(screen_number)

        lasso_model_path = lasso_model_path.replace('machine_learning', name_suffix)
        RF_model_path = RF_model_path.replace('machine_learning', name_suffix)
        NN_model_path = NN_model_path.replace('machine_learning', name_suffix)
        DT_model_path = DT_model_path.replace('machine_learning', name_suffix)
        SVM_model_path = SVM_model_path.replace('machine_learning', name_suffix)

    with open(lasso_model_path, 'rb') as handle:
            lasso_model = pickle.load(handle)
    with open(RF_model_path, 'rb') as handle:
            RF_model = pickle.load(handle)
    with open(NN_model_path, 'rb') as handle:
            NN_model = pickle.load(handle)
    with open(DT_model_path, 'rb') as handle:
            DT_model = pickle.load(handle)
    with open(SVM_model_path, 'rb') as handle:
            SVM_model = pickle.load(handle)

    # split training features and target values
    unirep64_feats = 128 # excluding physiochemical features
    X_df = embeddings_df.iloc[:,0:unirep64_feats].dropna(axis=1)#
    y_df = embeddings_df.copy().iloc[:, [-1]]

    # create train-test-splits
    test_size = 0.20

    r2_lasso_list = []
    r2_RF_list = []
    r2_NN_list=[]
    r2_DT_list = []
    r2_SVM_list = []

    r2_lasso_scram_list = []
    r2_RF_scram_list = []
    r2_NN_scram_list = []
    r2_DT_scram_list = []
    r2_SVM_scram_list = []

    for trial in range(ntrials):
        if trial % 5 == 0:
            print(trial)

        random_state = trial
        X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                            test_size=test_size,
                                                            random_state=random_state)
        X_scrambled_df = embeddings_scrambed_df.iloc[:,0:unirep64_feats].dropna(axis=1)#
        y_scrambled_df = embeddings_scrambed_df.copy().iloc[:, [-1]]
        X_scrambled_train = pd.concat([X_train, X_scrambled_df], axis=1).dropna().iloc[:,unirep64_feats:]
        X_scrambled_test = pd.concat([X_test, X_scrambled_df], axis=1).dropna().iloc[:,unirep64_feats:]
        y_scrambled_train = pd.concat([y_train, y_scrambled_df], axis=1).dropna().iloc[:,1:]
        y_scrambled_test = pd.concat([y_test, y_scrambled_df], axis=1).dropna().iloc[:,1:]

        r2_lasso, r2_RF, r2_NN, r2_DT, r2_SVM, r2_lasso_scram, r2_RF_scram, r2_NN_scram, r2_DT_scram,r2_SVM_scram = scramble_predictions(X_train, X_test, y_train, y_test,
                            X_scrambled_train, X_scrambled_test,
                            y_scrambled_train, y_scrambled_test,
                            lasso_model, RF_model,NN_model, DT_model, SVM_model)

        r2_lasso_list.append(r2_lasso)
        r2_RF_list.append(r2_RF)
        r2_NN_list.append(r2_NN)
        r2_DT_list.append(r2_DT)
        r2_SVM_list.append(r2_SVM)

        r2_lasso_scram_list.append(r2_lasso_scram)
        r2_RF_scram_list.append(r2_RF_scram)
        r2_NN_scram_list.append(r2_NN_scram)
        r2_DT_scram_list.append(r2_DT_scram)
        r2_SVM_scram_list.append(r2_SVM_scram)


    results_dict = {"r2_lasso": r2_lasso_list,
               "r2_RF": r2_RF_list,
               "r2_NN": r2_NN_list,
               "r2_DT": r2_DT_list,
               "r2_SVM": r2_SVM_list,
               "r2_lasso_scram": r2_lasso_scram_list,
               "r2_RF_scram": r2_RF_scram_list,
               "r2_NN_scram": r2_NN_scram_list,
               "r2_DT_scram_list": r2_DT_scram_list,
               "r2_SVM_scram_list": r2_SVM_scram_list}

    results_df = pd.DataFrame(results_dict)
    if screen_number==1:
        results_df.to_csv('../data/machine_learning/regression_scramble_comparison.csv')
    elif screen_number==2:
        results_df.to_csv('../data/machine_learning_screen2/regression_scramble_comparison.csv')
    elif screen_number==3:
        results_df.to_csv('../data/machine_learning_screen3/regression_scramble_comparison.csv')
    elif screen_number==4:
        results_df.to_csv('../data/machine_learning_screen4/regression_scramble_comparison.csv')

    lasso_model_stats = []
    RF_model_stats = []
    NN_model_stats = []
    DT_model_stats = []
    SVM_model_stats = []
    lasso_model_stats.append(stats.wilcoxon(r2_lasso_list, r2_lasso_scram_list)[1])
    RF_model_stats.append(stats.wilcoxon(r2_RF_list, r2_RF_scram_list)[1])
    NN_model_stats.append(stats.wilcoxon(r2_NN_list, r2_NN_scram_list)[1])
    DT_model_stats.append(stats.wilcoxon(r2_DT_list, r2_DT_scram_list)[1])
    SVM_model_stats.append(stats.wilcoxon(r2_SVM_list, r2_SVM_scram_list)[1])

    results_stats_dict = {"lasso_model_stats": lasso_model_stats,
                         "RF_model_stats": RF_model_stats,
                         "NN_model_stats": NN_model_stats,
                         "DT_model_stats": DT_model_stats,
                         "SVM_model_stats": SVM_model_stats}
    results_stats_df = pd.DataFrame(results_stats_dict)
    if screen_number==1:
        results_stats_df.to_csv('../data/machine_learning/regression_scramble_comparison_stats.csv')
    elif screen_number==2:
        results_stats_df.to_csv('../data/machine_learning_screen2/regression_scramble_comparison_stats.csv')
    elif screen_number==3:
        results_stats_df.to_csv('../data/machine_learning_screen3/regression_scramble_comparison_stats.csv')
    elif screen_number==4:
        results_stats_df.to_csv('../data/machine_learning_screen4/regression_scramble_comparison_stats.csv')

if __name__ == "__main__":
    main()
