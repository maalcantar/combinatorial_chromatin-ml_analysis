#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on November 13 13:30:13 2023
@author: alcantar
example run: python 10_context_aware_ml_models.py
"""
# activate virtual enviroment before running script
# source activate combicr_ml

import numpy as np
import pandas as pd
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
def train_pred_lasso_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path, out_figs_path,
                           save=False, context_aware=False):

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

    if context_aware:
        return(y_test, y_pred_testing)

def train_pred_random_forest_regressor_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path, out_figs_path,
                           save=False, context_aware=False):

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

    if context_aware:
        return(y_test, y_pred_testing)

def train_pred_KNN_regressor_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path, out_figs_path,
                           save=False, context_aware=False):

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

    if context_aware:
        return(y_test, y_pred_testing)

def train_pred_decision_tree_regressor_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path, out_figs_path,
                           save=False, context_aware=False):

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

    if context_aware:
        return(y_test, y_pred_testing)

def train_pred_support_vector_regressor_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path, out_figs_path,
                           save=False, context_aware=False):

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

    if context_aware:
        return(y_test, y_pred_testing)


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


def plot_context_results(y_test_true, ypred,
                         output_fig_path,output_data_path,
                         model_name, save=False):

    '''
    plot ml results by color coding each data point. also, return a dataframe with the ml prediction
    results which is ordered by screen number / context

    PARAMETERS
    --------------------
    y_test_true: pandas dataframe
        dataframe with measured normalized fluorescence values
    ypred: numpy array
        numpy array with predicted normalized fluorescence values
    output_fig_path: str
        path to output folder for figures
    output_data_path: str
        path to output folder for data
    model_name:str
        name of model to use in output name (e.g., lasso, knn, random forest)
    save: boolean
        indicate whether to save figures / data

    RETURNS
    --------------------
    None
    '''

    y_true_pred = pd.DataFrame(y_test_true)
    y_true_pred['flow_seq_score_pred'] = ypred

    cr_combo_context_list = []
    for cr_combo in y_true_pred.index:
        cr_combo_context = cr_combo.split('_')[-1]
        cr_combo_context_list.append(cr_combo_context)
    y_true_pred['context'] = cr_combo_context_list

    y_true_pred

    y_true_pred_screen1 = y_true_pred[y_true_pred.index.str.contains('_1')]
    y_true_pred_screen2 = y_true_pred[y_true_pred.index.str.contains('_2')]
    y_true_pred_screen3 = y_true_pred[y_true_pred.index.str.contains('_3')]
    y_true_pred_screen4 = y_true_pred[y_true_pred.index.str.contains('_4')]
    y_true_pred_ordered_df = pd.concat([y_true_pred_screen1, y_true_pred_screen2,
                                       y_true_pred_screen3, y_true_pred_screen4])

    pearsonR_train_screen1 = round(stats.pearsonr(list(y_true_pred_screen1['flow_seq_score'].values.ravel()), list(y_true_pred_screen1['flow_seq_score_pred'].values.ravel()))[0]**2,3)
    spearmanR_train_screen1 = round(stats.spearmanr(list(y_true_pred_screen1['flow_seq_score'].values.ravel()), list(y_true_pred_screen1['flow_seq_score_pred'].values.ravel()))[0],3)
    R2_train_screen1 = round(r2_score(list(y_true_pred_screen1['flow_seq_score']), y_true_pred_screen1['flow_seq_score_pred']),3)

    pearsonR_train_screen2 = round(stats.pearsonr(list(y_true_pred_screen2['flow_seq_score'].values.ravel()), list(y_true_pred_screen2['flow_seq_score_pred'].values.ravel()))[0]**2,3)
    spearmanR_train_screen2 = round(stats.spearmanr(list(y_true_pred_screen2['flow_seq_score'].values.ravel()), list(y_true_pred_screen2['flow_seq_score_pred'].values.ravel()))[0],3)
    R2_train_screen2 = round(r2_score(list(y_true_pred_screen2['flow_seq_score']), y_true_pred_screen2['flow_seq_score_pred']),3)

    pearsonR_train_screen3 = round(stats.pearsonr(list(y_true_pred_screen3['flow_seq_score'].values.ravel()), list(y_true_pred_screen3['flow_seq_score_pred'].values.ravel()))[0]**2,3)
    spearmanR_train_screen3 = round(stats.spearmanr(list(y_true_pred_screen3['flow_seq_score'].values.ravel()), list(y_true_pred_screen3['flow_seq_score_pred'].values.ravel()))[0],3)
    R2_train_screen3 = round(r2_score(list(y_true_pred_screen3['flow_seq_score']), y_true_pred_screen3['flow_seq_score_pred']),3)

    pearsonR_train_screen4 = round(stats.pearsonr(list(y_true_pred_screen4['flow_seq_score'].values.ravel()), list(y_true_pred_screen4['flow_seq_score_pred'].values.ravel()))[0]**2,3)
    spearmanR_train_screen4 = round(stats.spearmanr(list(y_true_pred_screen4['flow_seq_score'].values.ravel()), list(y_true_pred_screen4['flow_seq_score_pred'].values.ravel()))[0],3)
    R2_train_screen4 = round(r2_score(list(y_true_pred_screen4['flow_seq_score']), y_true_pred_screen4['flow_seq_score_pred']),3)

    pearsonR_train_all = round(stats.pearsonr(list(y_true_pred_ordered_df['flow_seq_score'].values.ravel()), list(y_true_pred_ordered_df['flow_seq_score_pred'].values.ravel()))[0]**2,3)
    spearmanR_train_all = round(stats.spearmanr(list(y_true_pred_ordered_df['flow_seq_score'].values.ravel()), list(y_true_pred_ordered_df['flow_seq_score_pred'].values.ravel()))[0],3)
    R2_train_all = round(r2_score(list(y_true_pred_ordered_df['flow_seq_score']), y_true_pred_ordered_df['flow_seq_score_pred']),3)

    plt.figure(dpi=400)
    sns.set(rc={'figure.figsize':(9,8)})
    sns.set_style("white")
    custom_palette = ["#000000", "#B6579A", "#7D8BC0", "#F0B584"]

    # Set the custom palette
    sns.set_palette(custom_palette)
    sns.scatterplot(data=y_true_pred_ordered_df, x='flow_seq_score',
                    y='flow_seq_score_pred', hue='context',
                    palette=custom_palette, s=30, linewidth=0, alpha=0.80,
                    marker='o')

    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.xlabel('Position', fontsize=16)
    plt.ylabel('Occupancy', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.text(1.1, 0.85, 'screen 1')
    plt.text(1.1, 0.81, f'Pearson r$^2$: {str(pearsonR_train_screen1)}')
    plt.text(1.1, 0.77, f'Spearman $\\rho$: {str(spearmanR_train_screen1)}')
    plt.text(1.1, 0.73, f'R$^2$: {str(R2_train_screen1)}')

    plt.text(1.1, 0.65, 'screen 2')
    plt.text(1.1, 0.61, f'Pearson r$^2$: {str(pearsonR_train_screen2)}')
    plt.text(1.1, 0.57, f'Spearman $\\rho$: {str(spearmanR_train_screen2)}')
    plt.text(1.1, 0.53, f'R$^2$: {str(R2_train_screen2)}')

    plt.text(1.1, 0.45, 'screen 3')
    plt.text(1.1, 0.41, f'Pearson r$^2$: {str(pearsonR_train_screen3)}')
    plt.text(1.1, 0.37, f'Spearman $\\rho$: {str(spearmanR_train_screen3)}')
    plt.text(1.1, 0.33, f'R$^2$: {str(R2_train_screen3)}')

    plt.text(1.1, 0.25, 'screen 4')
    plt.text(1.1, 0.21, f'Pearson r$^2$: {str(pearsonR_train_screen4)}')
    plt.text(1.1, 0.17, f'Spearman $\\rho$: {str(spearmanR_train_screen4)}')
    plt.text(1.1, 0.13, f'R$^2$: {str(R2_train_screen4)}')

    plt.text(1.1, 0.1, 'all')
    plt.text(1.1, 0.07, f'Pearson r$^2$: {str(pearsonR_train_all)}')
    plt.text(1.1, 0.04, f'Spearman $\\rho$: {str(spearmanR_train_all)}')
    plt.text(1.1, 0.01, f'R$^2$: {str(R2_train_all)}')

    if save:
        out_fig = output_fig_path + model_name +'_context_aware_results'
        out_data = output_data_path + model_name +'_context_aware_results'
        plt.savefig(out_fig +'.png' , bbox_inches='tight', dpi=400)
        plt.savefig(out_fig +'.pdf' , bbox_inches='tight', dpi=400)
        plt.savefig(out_fig +'.svg' , bbox_inches='tight', dpi=400)

        y_true_pred_screen1.to_csv(out_data + '_screen1.csv')
        y_true_pred_screen2.to_csv(out_data + '_screen2.csv')
        y_true_pred_screen3.to_csv(out_data + '_screen3.csv')
        y_true_pred_screen4.to_csv(out_data + '_screen4.csv')
        y_true_pred_ordered_df.to_csv(out_data+'_all.csv')

def main():
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

    # split training features and target values
    unirep64_feats = 132 # excluding physiochemical features
    X_df = screen_results_all.iloc[:,0:unirep64_feats].dropna(axis=1)
    y_df = screen_results_all.copy().iloc[:, [-1]]

    # create train-test-splits
    test_size = 0.20
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                        test_size=test_size,
                                                        random_state=random_state)

    data_dir_prefix = '../data/machine_learning_context_aware/'
    figs_dir_prefix = '../figs/machine_learning_context_aware/'

    lasso_output_data_path = data_dir_prefix + 'lasso_regression/'
    lasso_output_figs_path = figs_dir_prefix + 'lasso_regression/'
    # create ouput folders if needed
    make_dir(lasso_output_data_path)
    make_dir(lasso_output_figs_path)
    print('creating lasso model')
    y_test_true_lasso, ypred_lasso = train_pred_lasso_model(X_train, X_test,
                           y_train, y_test,
                           out_data_path=lasso_output_data_path,
                           out_figs_path=lasso_output_figs_path, save=True,
                           context_aware=True)

    #random forest regressor
    random_forest_output_data_path = data_dir_prefix + 'random_forest_regression/'
    random_forest_output_figs_path = figs_dir_prefix + 'random_forest_regression/'
    print(random_forest_output_data_path)
    print(random_forest_output_figs_path)
    make_dir(random_forest_output_data_path)
    make_dir(random_forest_output_figs_path)
    print('creating random forest model')
    y_test_true_rf, ypred_rf  = train_pred_random_forest_regressor_model(X_train, X_test,
                           y_train, y_test,
                           random_forest_output_data_path, random_forest_output_figs_path,
                           save=True, context_aware=True)
    print('rf results')

    # nearest neighbots regressor
    nearest_neighbors_output_data_path = data_dir_prefix + 'nearest_neighbors_regression/'
    nearest_neighbors_output_figs_path = figs_dir_prefix + 'nearest_neighbors_regression/'
    make_dir(nearest_neighbors_output_data_path)
    make_dir(nearest_neighbors_output_figs_path)
    print('creating nearest neighbors model')
    y_test_true_knn, ypred_knn = train_pred_KNN_regressor_model(X_train, X_test,
                           y_train, y_test,
                           nearest_neighbors_output_data_path, nearest_neighbors_output_figs_path,
                           save=True, context_aware=True)

    # decision tree regressor
    decision_tree_output_data_path = data_dir_prefix + 'decision_tree_regression/'
    decision_tree_output_figs_path = figs_dir_prefix + 'decision_tree_regression/'
    make_dir(decision_tree_output_data_path)
    make_dir(decision_tree_output_figs_path)
    print('creating decision tree model')
    y_test_true_dt, ypred_dt = train_pred_decision_tree_regressor_model(X_train, X_test,
                           y_train, y_test,
                           decision_tree_output_data_path, decision_tree_output_figs_path,
                           save=True, context_aware=True)

   # support vector regressor
    support_vector_reg_output_data_path = data_dir_prefix + 'support_vector_regression/'
    support_vector_reg_output_figs_path = figs_dir_prefix + 'support_vector_regression/'
    print(support_vector_reg_output_data_path)
    print(support_vector_reg_output_figs_path)
    make_dir(support_vector_reg_output_data_path)
    make_dir(support_vector_reg_output_figs_path)
    print('creating support vector model')
    y_test_true_svm, ypred_svm = train_pred_support_vector_regressor_model(X_train, X_test,
                           y_train, y_test,
                           support_vector_reg_output_data_path, support_vector_reg_output_figs_path,
                           save=True, context_aware=True)

    make_dir(data_dir_prefix)
    make_dir(figs_dir_prefix)

    plot_context_results(y_test_true_lasso, ypred_lasso,
          figs_dir_prefix,data_dir_prefix, model_name='lasso',
          save=True)

    plot_context_results(y_test_true_rf, ypred_rf,
          figs_dir_prefix,data_dir_prefix, model_name='random_forest',
          save=True)

    plot_context_results(y_test_true_knn, ypred_knn,
          figs_dir_prefix,data_dir_prefix, model_name='k_nearest_neighbors',
          save=True)

    plot_context_results(y_test_true_dt, ypred_dt,
          figs_dir_prefix,data_dir_prefix, model_name='decision_tree',
          save=True)

    plot_context_results(y_test_true_svm, ypred_svm,
          figs_dir_prefix,data_dir_prefix, model_name='support_vector',
          save=True)

if __name__ == "__main__":
    main()
