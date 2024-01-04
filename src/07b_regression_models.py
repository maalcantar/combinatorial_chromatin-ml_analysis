#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 19 11:55:04 2023
@author: alcantar
example run: python 07b_regression_models.py -i ../data/embedding_dataframes_clean/unirep64_final_embeddings.csv
example run 2: python 07b_regression_models.py -i ../data/embedding_dataframes_screen2_clean/unirep64_final_screen2_embeddings.csv -s 2
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
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel

from scipy import stats
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import seaborn as sns

import os
import pickle
import argparse

def train_pred_lasso_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path, out_figs_path,
                           save=False):

    '''
    make linear regression model with L1 regularization.

    PARAMETERS
    --------------------
    X_train: pandas dataframe
        dataframe with training features
    X_test: pandas dataframe
        dataframe with testing features
    y_train: pandas dataframe
        dataframe with training target values
    Y_test: pandas dataframe
        dataframe with testing target values
    out_data_path: string
        output path for data (i.e., model pickle file and csv with train/test values)
    out_figs_path: string
        output path for all figures
    save: boolean
        indicate whether figures, model, and dataframes will be saved

    RETURNS
    --------------------
    none

    '''

    # initialize lasso model
    reg = Lasso()

    # coeficient size penalty
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}

    # initialize grid search object
    grid_search = GridSearchCV(reg, param_grid, cv=5)

    # conduct grid search with training data
    grid_search.fit(X_train, y_train)

    # print best parameters
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)
    model =  grid_search.best_estimator_

    # save model as pickle file
    if save:
        pickle_path = out_data_path + 'lasso_regression_model.pkl'
        with open(pickle_path, 'wb') as f:
             pickle.dump(model, f)

    # make predictions on training and testing data and obtain statistics
    y_pred_training = model.predict(X_train)
    pearsonR_train = round(stats.pearsonr(list(y_train['flow_seq_score'].values.ravel()), y_pred_training)[0]**2,3)
    spearmanR_train = round(stats.spearmanr(list(y_train['flow_seq_score'].values.ravel()), y_pred_training)[0],3)
    R2_train = round(r2_score(list(y_train['flow_seq_score']), y_pred_training),3)

    y_pred_testing = model.predict(X_test)
    pearsonR_test = round(stats.pearsonr(list(y_test['flow_seq_score'].values.ravel()), y_pred_testing)[0]**2,3)
    spearmanR_test = round(stats.spearmanr(list(y_test['flow_seq_score'].values.ravel()), y_pred_testing)[0],3)
    R2_test = round(r2_score(list(y_test['flow_seq_score']), y_pred_testing),3)


    # plot training data
    fig_train = plt.figure(dpi=400)
    ax_train = fig_train.add_subplot(1, 1, 1)
    ax_train.scatter(y_train, y_pred_training)
    ax_train.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR_train)}')
    ax_train.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR_train)}')
    ax_train.text(0.1, 0.80, f'R$^2$: {str(R2_train)}')

    ax_train.set_xlim([0.0, 1.05])
    ax_train.set_ylim([0.0, 1.05])
    ax_train.set_xlabel('Measured score')
    ax_train.set_ylabel('Predicted score')
    plt.title('Lasso regression model: training results')

    # save training data figure and csv
    if save:
        out_training_fig = out_figs_path + 'lasso_regression_training'
        plt.savefig(out_training_fig +'.png')
        plt.savefig(out_training_fig +'.pdf')
        plt.savefig(out_training_fig +'.svg')

        train_dataframe = pd.DataFrame(list(zip(y_train['flow_seq_score'], y_pred_training)),
               columns =['y_train', 'y_pred_train']).rename(index=dict(enumerate(y_train.index)))

        out_training_df_path = out_data_path + 'lasso_regression_training.csv'
        train_dataframe.to_csv(out_training_df_path)

    # plot testing data
    fig_test = plt.figure(dpi=400)
    ax_test = fig_test.add_subplot(1, 1, 1)
    ax_test.scatter(y_test, y_pred_testing)
    ax_test.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR_test)}')
    ax_test.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR_test)}')
    ax_test.text(0.1, 0.80, f'R$^2$: {str(R2_test)}')

    ax_test.set_xlim([0.0, 1.05])
    ax_test.set_ylim([0.0, 1.05])
    ax_test.set_xlabel('Measured score')
    ax_test.set_ylabel('Predicted score')
    plt.title('Lasso regression model: test results')

    # save testing data figures and csv
    if save:
        out_training_fig = out_figs_path + 'lasso_regression_testing'
        plt.savefig(out_training_fig +'.png')
        plt.savefig(out_training_fig +'.pdf')
        plt.savefig(out_training_fig +'.svg')

        test_dataframe = pd.DataFrame(list(zip(y_test['flow_seq_score'], y_pred_testing)),
               columns =['y_test', 'y_pred_test']).rename(index=dict(enumerate(y_test.index)))

        out_testing_df_path = out_data_path + 'lasso_regression_testing.csv'
        test_dataframe.to_csv(out_testing_df_path)

def train_pred_random_forest_regressor_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path, out_figs_path,
                           save=False):

    '''
    make random forest regression model

    PARAMETERS
    --------------------
    X_train: pandas dataframe
        dataframe with training features
    X_test: pandas dataframe
        dataframe with testing features
    y_train: pandas dataframe
        dataframe with training target values
    Y_test: pandas dataframe
        dataframe with testing target values
    out_data_path: string
        output path for data (i.e., model pickle file and csv with train/test values)
    out_figs_path: string
        output path for all figures
    save: boolean
        indicate whether figures, model, and dataframes will be saved

    RETURNS
    --------------------
    none

    '''

    # paramters for random forest model: number of trees and max number of branches
    param_grid = {'n_estimators': [50, 100, 200],
                  'max_depth': [5, 10, 15]}

    # initialize random forest regressor model
    regr = RandomForestRegressor(random_state=42)

    # initialize gridsearch object
    grid_search = GridSearchCV(regr, param_grid, cv=5)

    # fit the gridsearch object
    grid_search.fit(X_train, y_train.values.ravel())

    # print the best parameters
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    model = grid_search.best_estimator_

    # save model
    if save:
        pickle_path = out_data_path + 'random_forest_regression_model.pkl'
        with open(pickle_path, 'wb') as f:
             pickle.dump(model, f)

    # make predictions on training and testing data and save statistics
    y_pred_training = model.predict(X_train)
    pearsonR_train = round(stats.pearsonr(list(y_train['flow_seq_score'].values.ravel()), y_pred_training)[0]**2,3)
    spearmanR_train = round(stats.spearmanr(list(y_train['flow_seq_score'].values.ravel()), y_pred_training)[0],3)
    R2_train = round(r2_score(list(y_train['flow_seq_score']), y_pred_training),3)

    y_pred_testing = model.predict(X_test)
    pearsonR_test = round(stats.pearsonr(list(y_test['flow_seq_score'].values.ravel()), y_pred_testing)[0]**2,3)
    spearmanR_test = round(stats.spearmanr(list(y_test['flow_seq_score'].values.ravel()), y_pred_testing)[0],3)
    R2_test = round(r2_score(list(y_test['flow_seq_score']), y_pred_testing),3)


    # plot training data
    fig_train = plt.figure(dpi=400)
    ax_train = fig_train.add_subplot(1, 1, 1)
    ax_train.scatter(y_train, y_pred_training)
    ax_train.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR_train)}')
    ax_train.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR_train)}')
    ax_train.text(0.1, 0.80, f'R$^2$: {str(R2_train)}')

    ax_train.set_xlim([0.0, 1.05])
    ax_train.set_ylim([0.0, 1.05])
    ax_train.set_xlabel('Measured score')
    ax_train.set_ylabel('Predicted score')
    plt.title('Random forest regression model: training results')

    # save training figures and csv
    if save:
        out_training_fig = out_figs_path + 'random_forest_regression_training'
        plt.savefig(out_training_fig +'.png')
        plt.savefig(out_training_fig +'.pdf')
        plt.savefig(out_training_fig +'.svg')

        train_dataframe = pd.DataFrame(list(zip(y_train['flow_seq_score'], y_pred_training)),
               columns =['y_train', 'y_pred_train']).rename(index=dict(enumerate(y_train.index)))

        out_training_df_path = out_data_path + 'random_forest_regression_training.csv'
        train_dataframe.to_csv(out_training_df_path)

    # plot testing data
    fig_test = plt.figure(dpi=400)
    ax_test = fig_test.add_subplot(1, 1, 1)
    ax_test.scatter(y_test, y_pred_testing)
    ax_test.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR_test)}')
    ax_test.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR_test)}')
    ax_test.text(0.1, 0.80, f'R$^2$: {str(R2_test)}')

    ax_test.set_xlim([0.0, 1.05])
    ax_test.set_ylim([0.0, 1.05])
    ax_test.set_xlabel('Measured score')
    ax_test.set_ylabel('Predicted score')
    plt.title('Random forest regression model: test results')

    # save testing figures and csv
    if save:
        out_training_fig = out_figs_path + 'random_forest_regression_testing'
        plt.savefig(out_training_fig +'.png')
        plt.savefig(out_training_fig +'.pdf')
        plt.savefig(out_training_fig +'.svg')

        test_dataframe = pd.DataFrame(list(zip(y_test['flow_seq_score'], y_pred_testing)),
               columns =['y_test', 'y_pred_test']).rename(index=dict(enumerate(y_test.index)))

        out_testing_df_path = out_data_path + 'random_forest_regression_testing.csv'
        test_dataframe.to_csv(out_testing_df_path)

def train_pred_KNN_regressor_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path, out_figs_path,
                           save=False):

    '''
    make nearest neighbors regression model

    PARAMETERS
    --------------------
    X_train: pandas dataframe
        dataframe with training features
    X_test: pandas dataframe
        dataframe with testing features
    y_train: pandas dataframe
        dataframe with training target values
    Y_test: pandas dataframe
        dataframe with testing target values
    out_data_path: string
        output path for data (i.e., model pickle file and csv with train/test values)
    out_figs_path: string
        output path for all figures
    save: boolean
        indicate whether figures, model, and dataframes will be saved

    RETURNS
    --------------------
    none

    '''

    # paramters for random forest model: number of trees and max number of branches
    param_grid = {'n_neighbors': [3, 5, 7],
              'weights': ['uniform', 'distance']}

    # initialize random forest regressor model
    regr = KNeighborsRegressor()

    # initialize gridsearch object
    grid_search = GridSearchCV(regr, param_grid, cv=5)

    # fit the gridsearch object
    grid_search.fit(X_train, y_train.values.ravel())

    # print the best parameters
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    model = grid_search.best_estimator_

    # save model
    if save:
        pickle_path = out_data_path + 'KNN_regression_model.pkl'
        with open(pickle_path, 'wb') as f:
             pickle.dump(model, f)

    # make predictions on training and testing data and save statistics
    y_pred_training = model.predict(X_train)
    pearsonR_train = round(stats.pearsonr(list(y_train['flow_seq_score'].values.ravel()), y_pred_training)[0]**2,3)
    spearmanR_train = round(stats.spearmanr(list(y_train['flow_seq_score'].values.ravel()), y_pred_training)[0],3)
    R2_train = round(r2_score(list(y_train['flow_seq_score']), y_pred_training),3)

    y_pred_testing = model.predict(X_test)
    pearsonR_test = round(stats.pearsonr(list(y_test['flow_seq_score'].values.ravel()), y_pred_testing)[0]**2,3)
    spearmanR_test = round(stats.spearmanr(list(y_test['flow_seq_score'].values.ravel()), y_pred_testing)[0],3)
    R2_test = round(r2_score(list(y_test['flow_seq_score']), y_pred_testing),3)


    # plot training data
    fig_train = plt.figure(dpi=400)
    ax_train = fig_train.add_subplot(1, 1, 1)
    ax_train.scatter(y_train, y_pred_training)
    ax_train.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR_train)}')
    ax_train.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR_train)}')
    ax_train.text(0.1, 0.80, f'R$^2$: {str(R2_train)}')

    ax_train.set_xlim([0.0, 1.05])
    ax_train.set_ylim([0.0, 1.05])
    ax_train.set_xlabel('Measured score')
    ax_train.set_ylabel('Predicted score')
    plt.title('KNN regression model: training results')

    # save training figures and csv
    if save:
        out_training_fig = out_figs_path + 'KNN_regression_training'
        plt.savefig(out_training_fig +'.png')
        plt.savefig(out_training_fig +'.pdf')
        plt.savefig(out_training_fig +'.svg')

        train_dataframe = pd.DataFrame(list(zip(y_train['flow_seq_score'], y_pred_training)),
               columns =['y_train', 'y_pred_train']).rename(index=dict(enumerate(y_train.index)))

        out_training_df_path = out_data_path + 'KNN_regression_training.csv'
        train_dataframe.to_csv(out_training_df_path)

    # plot testing data
    fig_test = plt.figure(dpi=400)
    ax_test = fig_test.add_subplot(1, 1, 1)
    ax_test.scatter(y_test, y_pred_testing)
    ax_test.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR_test)}')
    ax_test.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR_test)}')
    ax_test.text(0.1, 0.80, f'R$^2$: {str(R2_test)}')

    ax_test.set_xlim([0.0, 1.05])
    ax_test.set_ylim([0.0, 1.05])
    ax_test.set_xlabel('Measured score')
    ax_test.set_ylabel('Predicted score')
    plt.title('KNN regression model: test results')

    # save testing figures and csv
    if save:
        out_training_fig = out_figs_path + 'KNN_regression_testing'
        plt.savefig(out_training_fig +'.png')
        plt.savefig(out_training_fig +'.pdf')
        plt.savefig(out_training_fig +'.svg')

        test_dataframe = pd.DataFrame(list(zip(y_test['flow_seq_score'], y_pred_testing)),
               columns =['y_test', 'y_pred_test']).rename(index=dict(enumerate(y_test.index)))

        out_testing_df_path = out_data_path + 'KNN_regression_testing.csv'
        test_dataframe.to_csv(out_testing_df_path)

def train_pred_decision_tree_regressor_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path, out_figs_path,
                           save=False):

    '''
    make nearest neighbors regression model

    PARAMETERS
    --------------------
    X_train: pandas dataframe
        dataframe with training features
    X_test: pandas dataframe
        dataframe with testing features
    y_train: pandas dataframe
        dataframe with training target values
    Y_test: pandas dataframe
        dataframe with testing target values
    out_data_path: string
        output path for data (i.e., model pickle file and csv with train/test values)
    out_figs_path: string
        output path for all figures
    save: boolean
        indicate whether figures, model, and dataframes will be saved

    RETURNS
    --------------------
    none

    '''

    # paramters for random forest model: number of trees and max number of branches
    param_grid = {'max_depth': [1, 2, 3, 4, 5],
              'min_samples_split': [2, 3, 4, 5, 6]}

    # initialize random forest regressor model
    regr = DecisionTreeRegressor()

    # initialize gridsearch object
    grid_search = GridSearchCV(regr, param_grid, cv=5)

    # fit the gridsearch object
    grid_search.fit(X_train, y_train.values.ravel())

    # print the best parameters
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    model = grid_search.best_estimator_

    # save model
    if save:
        pickle_path = out_data_path + 'decision_tree_regression_model.pkl'
        with open(pickle_path, 'wb') as f:
             pickle.dump(model, f)

    # make predictions on training and testing data and save statistics
    y_pred_training = model.predict(X_train)
    pearsonR_train = round(stats.pearsonr(list(y_train['flow_seq_score'].values.ravel()), y_pred_training)[0]**2,3)
    spearmanR_train = round(stats.spearmanr(list(y_train['flow_seq_score'].values.ravel()), y_pred_training)[0],3)
    R2_train = round(r2_score(list(y_train['flow_seq_score']), y_pred_training),3)

    y_pred_testing = model.predict(X_test)
    pearsonR_test = round(stats.pearsonr(list(y_test['flow_seq_score'].values.ravel()), y_pred_testing)[0]**2,3)
    spearmanR_test = round(stats.spearmanr(list(y_test['flow_seq_score'].values.ravel()), y_pred_testing)[0],3)
    R2_test = round(r2_score(list(y_test['flow_seq_score']), y_pred_testing),3)


    # plot training data
    fig_train = plt.figure(dpi=400)
    ax_train = fig_train.add_subplot(1, 1, 1)
    ax_train.scatter(y_train, y_pred_training)
    ax_train.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR_train)}')
    ax_train.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR_train)}')
    ax_train.text(0.1, 0.80, f'R$^2$: {str(R2_train)}')

    ax_train.set_xlim([0.0, 1.05])
    ax_train.set_ylim([0.0, 1.05])
    ax_train.set_xlabel('Measured score')
    ax_train.set_ylabel('Predicted score')
    plt.title('Decision tree regression model: training results')

    # save training figures and csv
    if save:
        out_training_fig = out_figs_path + 'Decision_tree_regression_training'
        plt.savefig(out_training_fig +'.png')
        plt.savefig(out_training_fig +'.pdf')
        plt.savefig(out_training_fig +'.svg')

        train_dataframe = pd.DataFrame(list(zip(y_train['flow_seq_score'], y_pred_training)),
               columns =['y_train', 'y_pred_train']).rename(index=dict(enumerate(y_train.index)))

        out_training_df_path = out_data_path + 'decision_tree_regression_training.csv'
        train_dataframe.to_csv(out_training_df_path)

    # plot testing data
    fig_test = plt.figure(dpi=400)
    ax_test = fig_test.add_subplot(1, 1, 1)
    ax_test.scatter(y_test, y_pred_testing)
    ax_test.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR_test)}')
    ax_test.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR_test)}')
    ax_test.text(0.1, 0.80, f'R$^2$: {str(R2_test)}')

    ax_test.set_xlim([0.0, 1.05])
    ax_test.set_ylim([0.0, 1.05])
    ax_test.set_xlabel('Measured score')
    ax_test.set_ylabel('Predicted score')
    plt.title('decision tree regression model: test results')

    # save testing figures and csv
    if save:
        out_training_fig = out_figs_path + 'decision_tree_regression_testing'
        plt.savefig(out_training_fig +'.png')
        plt.savefig(out_training_fig +'.pdf')
        plt.savefig(out_training_fig +'.svg')

        test_dataframe = pd.DataFrame(list(zip(y_test['flow_seq_score'], y_pred_testing)),
               columns =['y_test', 'y_pred_test']).rename(index=dict(enumerate(y_test.index)))

        out_testing_df_path = out_data_path + 'decision_tree_regression_testing.csv'
        test_dataframe.to_csv(out_testing_df_path)

def train_pred_support_vector_regressor_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path, out_figs_path,
                           save=False):

    '''
    make nearest neighbors regression model

    PARAMETERS
    --------------------
    X_train: pandas dataframe
        dataframe with training features
    X_test: pandas dataframe
        dataframe with testing features
    y_train: pandas dataframe
        dataframe with training target values
    Y_test: pandas dataframe
        dataframe with testing target values
    out_data_path: string
        output path for data (i.e., model pickle file and csv with train/test values)
    out_figs_path: string
        output path for all figures
    save: boolean
        indicate whether figures, model, and dataframes will be saved

    RETURNS
    --------------------
    none

    '''

    # paramters for support vector model: number of trees and max number of branches
    param_grid = {
                    'svr__kernel': ['linear', 'rbf', 'poly'],
                    'svr__C': [0.1, 1, 10],
                    'svr__epsilon': [0.01, 0.1, 0.2],
                    'feature_selection__estimator__alpha': [0.001, 0.01, 0.1, 1.0]
                }

    # Create a Support Vector Machine regressor object
    svm = SVR()

    # Create a Lasso feature selection estimator
    lasso = Lasso()

    # Create a pipeline that includes feature selection and SVR
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(lasso)),
        ('svr', svm)
    ])

    # initialize gridsearch object
    grid_search = GridSearchCV(pipeline,
                                param_grid,
                                cv=5)

    # fit the gridsearch object
    grid_search.fit(X_train, y_train.values.ravel())

    # print the best parameters
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    model = grid_search.best_estimator_

    # save model
    if save:
        pickle_path = out_data_path + 'support_vector_regression_model.pkl'
        with open(pickle_path, 'wb') as f:
             pickle.dump(model, f)

    # make predictions on training and testing data and save statistics
    y_pred_training = model.predict(X_train)
    pearsonR_train = round(stats.pearsonr(list(y_train['flow_seq_score'].values.ravel()), y_pred_training)[0]**2,3)
    spearmanR_train = round(stats.spearmanr(list(y_train['flow_seq_score'].values.ravel()), y_pred_training)[0],3)
    R2_train = round(r2_score(list(y_train['flow_seq_score']), y_pred_training),3)

    y_pred_testing = model.predict(X_test)
    pearsonR_test = round(stats.pearsonr(list(y_test['flow_seq_score'].values.ravel()), y_pred_testing)[0]**2,3)
    spearmanR_test = round(stats.spearmanr(list(y_test['flow_seq_score'].values.ravel()), y_pred_testing)[0],3)
    R2_test = round(r2_score(list(y_test['flow_seq_score']), y_pred_testing),3)


    # plot training data
    fig_train = plt.figure(dpi=400)
    ax_train = fig_train.add_subplot(1, 1, 1)
    ax_train.scatter(y_train, y_pred_training)
    ax_train.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR_train)}')
    ax_train.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR_train)}')
    ax_train.text(0.1, 0.80, f'R$^2$: {str(R2_train)}')

    ax_train.set_xlim([0.0, 1.05])
    ax_train.set_ylim([0.0, 1.05])
    ax_train.set_xlabel('Measured score')
    ax_train.set_ylabel('Predicted score')
    plt.title('Support vector regression model: training results')

    # save training figures and csv
    if save:
        out_training_fig = out_figs_path + 'support_vector_regression_training'
        plt.savefig(out_training_fig +'.png')
        plt.savefig(out_training_fig +'.pdf')
        plt.savefig(out_training_fig +'.svg')

        train_dataframe = pd.DataFrame(list(zip(y_train['flow_seq_score'], y_pred_training)),
               columns =['y_train', 'y_pred_train']).rename(index=dict(enumerate(y_train.index)))

        out_training_df_path = out_data_path + 'support_vector_regression_training.csv'
        train_dataframe.to_csv(out_training_df_path)

    # plot testing data
    fig_test = plt.figure(dpi=400)
    ax_test = fig_test.add_subplot(1, 1, 1)
    ax_test.scatter(y_test, y_pred_testing)
    ax_test.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR_test)}')
    ax_test.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR_test)}')
    ax_test.text(0.1, 0.80, f'R$^2$: {str(R2_test)}')

    ax_test.set_xlim([0.0, 1.05])
    ax_test.set_ylim([0.0, 1.05])
    ax_test.set_xlabel('Measured score')
    ax_test.set_ylabel('Predicted score')
    plt.title('support vector regression model: test results')

    # save testing figures and csv
    if save:
        out_training_fig = out_figs_path + 'support_vector_regression_testing'
        plt.savefig(out_training_fig +'.png')
        plt.savefig(out_training_fig +'.pdf')
        plt.savefig(out_training_fig +'.svg')

        test_dataframe = pd.DataFrame(list(zip(y_test['flow_seq_score'], y_pred_testing)),
               columns =['y_test', 'y_pred_test']).rename(index=dict(enumerate(y_test.index)))

        out_testing_df_path = out_data_path + 'support_vector_regression_testing.csv'
        test_dataframe.to_csv(out_testing_df_path)

def make_dir(dir_path):

    '''
    check if a directory exists. if it does not, create the directory.

    PARAMETERS
    --------------------
    dir_path: string
        path to directory

    RETURNS
    --------------------
    none

    '''
    CHECK_FOLDER = os.path.isdir(dir_path)
    if not CHECK_FOLDER:
        os.makedirs(dir_path)
        print(f"created folder : {dir_path}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to embeddings dataframe')
    parser.add_argument('-s', help='screen number')

    args = parser.parse_args()
    embeddings_path = args.i
    screen_number = str(args.s)

    data_dir_prefix = '../data/machine_learning/'
    figs_dir_prefix = '../figs/machine_learning/'

    if screen_number == '2':
        data_dir_prefix = data_dir_prefix.replace('machine_learning', 'machine_learning_screen2')
        figs_dir_prefix = figs_dir_prefix.replace('machine_learning', 'machine_learning_screen2')
    elif screen_number == '3':
        data_dir_prefix = data_dir_prefix.replace('machine_learning', 'machine_learning_screen3')
        figs_dir_prefix = figs_dir_prefix.replace('machine_learning', 'machine_learning_screen3')
    elif screen_number == '4':
        data_dir_prefix = data_dir_prefix.replace('machine_learning', 'machine_learning_screen4')
        figs_dir_prefix = figs_dir_prefix.replace('machine_learning', 'machine_learning_screen4')
    # load embeddings dataframe
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)

    # split training features and target values
    unirep64_feats = 128 # excluding physiochemical features
    X_df = embeddings_df.iloc[:,0:unirep64_feats].dropna(axis=1)#
    y_df = embeddings_df.copy().iloc[:, [-1]]

    # create train-test-splits
    test_size = 0.20
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                        test_size=test_size,
                                                        random_state=random_state)

    # lasso model
    lasso_output_data_path = data_dir_prefix + 'lasso_regression/'
    lasso_output_figs_path = figs_dir_prefix + 'lasso_regression/'
    # create ouput folders if needed
    make_dir(lasso_output_data_path)
    make_dir(lasso_output_figs_path)
    print('creating lasso model')
    test_run = train_pred_lasso_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path=lasso_output_data_path,
                           out_figs_path=lasso_output_figs_path, save=True)

   # random forest regressor
    random_forest_output_data_path = data_dir_prefix + 'random_forest_regression/'
    random_forest_output_figs_path = figs_dir_prefix + 'random_forest_regression/'
    print(random_forest_output_data_path)
    print(random_forest_output_figs_path)
    make_dir(random_forest_output_data_path)
    make_dir(random_forest_output_figs_path)
    print('creating random forest model')
    train_pred_random_forest_regressor_model(X_train, X_test,
                           y_train, y_test,
                           random_forest_output_data_path, random_forest_output_figs_path,
                           save=True)
   # nearest neighbots regressor
    nearest_neighbors_output_data_path = data_dir_prefix + 'nearest_neighbors_regression/'
    nearest_neighbors_output_figs_path = figs_dir_prefix + 'nearest_neighbors_regression/'
    make_dir(nearest_neighbors_output_data_path)
    make_dir(nearest_neighbors_output_figs_path)
    print('creating nearest neighbors model')
    train_pred_KNN_regressor_model(X_train, X_test,
                           y_train, y_test,
                           nearest_neighbors_output_data_path, nearest_neighbors_output_figs_path,
                           save=True)

   # decision tree regressor
    decision_tree_output_data_path = data_dir_prefix + 'decision_tree_regression/'
    decision_tree_output_figs_path = figs_dir_prefix + 'decision_tree_regression/'
    make_dir(decision_tree_output_data_path)
    make_dir(decision_tree_output_figs_path)
    print('creating decision tree model')
    train_pred_decision_tree_regressor_model(X_train, X_test,
                           y_train, y_test,
                           decision_tree_output_data_path, decision_tree_output_figs_path,
                           save=True)

   # support vector regressor
    support_vector_reg_output_data_path = data_dir_prefix + 'support_vector_regression/'
    support_vector_reg_output_figs_path = figs_dir_prefix + 'support_vector_regression/'
    print(support_vector_reg_output_data_path)
    print(support_vector_reg_output_figs_path)
    make_dir(support_vector_reg_output_data_path)
    make_dir(support_vector_reg_output_figs_path)
    print('creating random forest model')
    train_pred_support_vector_regressor_model(X_train, X_test,
                           y_train, y_test,
                           support_vector_reg_output_data_path, support_vector_reg_output_figs_path,
                           save=True)

if __name__ == "__main__":
    main()
