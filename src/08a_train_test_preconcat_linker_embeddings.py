#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on February 24 10:28:20 2023
@author: alcantar
example run: python 08a_train_test_preconcat_linker_embeddings.py
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
    model: sci-kit learn model
        best linear regression model

    '''

    # initialize lasso model
    reg = Lasso()

    # coeficient size penalty
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}

    # initialize grid search object
    grid_search = GridSearchCV(reg, param_grid, cv=5, n_jobs=6)

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

    return(model)

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
    model: sci-kit learn model
        best random forest model

    '''

    # paramters for random forest model: number of trees and max number of branches
    param_grid = {'n_estimators': [50, 100, 200],
                  'max_depth': [5, 10, 15]}

    # initialize random forest regressor model
    regr = RandomForestRegressor(random_state=42)

    # initialize gridsearch object
    grid_search = GridSearchCV(regr, param_grid, cv=5, n_jobs=6)

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

        return(model)

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
    model: sci-kit learn model
        best nearest neighbors model

    '''

    # paramters for random forest model: number of trees and max number of branches
    param_grid = {'n_neighbors': [3, 5, 7],
              'weights': ['uniform', 'distance']}

    # initialize random forest regressor model
    regr = KNeighborsRegressor()

    # initialize gridsearch object
    grid_search = GridSearchCV(regr, param_grid, cv=5, n_jobs=6)

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

    return(model)

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
    model: sci-kit learn model
        best decision tree model

    '''

    # paramters for random forest model: number of trees and max number of branches
    param_grid = {'max_depth': [1, 2, 3, 4, 5],
              'min_samples_split': [2, 3, 4, 5, 6]}

    # initialize random forest regressor model
    regr = DecisionTreeRegressor()

    # initialize gridsearch object
    grid_search = GridSearchCV(regr, param_grid, cv=5, n_jobs=6)

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

    return(model)

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
    return(model)

def train_all_models(embeddings_path, screen_number, scrambled):


    '''
    train models

    PARAMETERS
    --------------------
    embeddings_path: string
        path to amino acid embeddings
    screen_number: int
        screen number 1 or 2
    scrambled: boolean
        indicate whether embeddings are from scrambled proteins or not

    RETURNS
    --------------------
    lasso_model: sci-kit learn model
        best lasso model
    RF_model: sci-kit learn model
        best random forest model
    KNN_model: sci-kit learn model
        best nearest neighbors model
    DT_model: sci-kit learn model
        best decision tree model
    SVM_model: sci-kit learn model
        best support vector model

    '''

    # embeddings_path = '../data/embedding_dataframes_clean/unirep64_preconcat_final_embeddings.csv'
#     embeddings_path = '../data/embedding_dataframes_clean/unirep64_preconcat_scrambled_final_embeddings.csv'
#     screen_number = 1

    if not scrambled:
        data_dir_prefix = '../data/machine_learning_preconcat_linker/'
        figs_dir_prefix = '../figs/machine_learning_preconcat_linker/'
    else:
        data_dir_prefix = '../data/machine_learning_scrambled_preconcat_linker/'
        figs_dir_prefix = '../figs/machine_learning_scrambled_preconcat_linker/'

    if screen_number == '2':
        data_dir_prefix = data_dir_prefix.replace('machine_learning', 'machine_learning_screen2')
        figs_dir_prefix = figs_dir_prefix.replace('machine_learning', 'machine_learning_screen2')
    # load embeddings dataframe
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)

    # split training features and target values
    unirep64_feats = 64 # excluding physiochemical features
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
    lasso_model = train_pred_lasso_model(X_train, X_test,
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
    RF_model = train_pred_random_forest_regressor_model(X_train, X_test,
                           y_train, y_test,
                           random_forest_output_data_path, random_forest_output_figs_path,
                           save=True)

    # nearest neighbots regressor
    nearest_neighbors_output_data_path = data_dir_prefix + 'nearest_neighbors_regression/'
    nearest_neighbors_output_figs_path = figs_dir_prefix + 'nearest_neighbors_regression/'
    make_dir(nearest_neighbors_output_data_path)
    make_dir(nearest_neighbors_output_figs_path)
    print('creating nearest neighbors model')
    KNN_model = train_pred_KNN_regressor_model(X_train, X_test,
                           y_train, y_test,
                           nearest_neighbors_output_data_path, nearest_neighbors_output_figs_path,
                           save=True)

    # decision tree regressor
    decision_tree_output_data_path = data_dir_prefix + 'decision_tree_regression/'
    decision_tree_output_figs_path = figs_dir_prefix + 'decision_tree_regression/'
    make_dir(decision_tree_output_data_path)
    make_dir(decision_tree_output_figs_path)
    print('creating decision tree model')
    DT_model = train_pred_decision_tree_regressor_model(X_train, X_test,
                           y_train, y_test,
                           decision_tree_output_data_path, decision_tree_output_figs_path,
                           save=True)

   # support vector regressor
    support_vector_output_data_path = data_dir_prefix + 'support_vector_regression/'
    support_vector_output_figs_path = figs_dir_prefix + 'support_vector_regression/'
    make_dir(support_vector_output_data_path)
    make_dir(support_vector_output_figs_path)
    print('creating support vector model')
    SVM_model = train_pred_support_vector_regressor_model(X_train, X_test,
                          y_train, y_test,
                          support_vector_output_data_path, support_vector_output_figs_path,
                          save=True)

    return(lasso_model, RF_model, KNN_model, DT_model, SVM_model)

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

    return(r2_lasso, r2_RF, r2_NN, r2_DT, r2_SVM, r2_lasso_scram, r2_RF_scram, r2_NN_scram, r2_DT_scram, r2_SVM_scram)

def main():
    embeddings_path = '../data/embedding_dataframes_clean/unirep64_preconcat_linker_final_embeddings.csv'
    screen_number = 1
    scrambled = False

    lasso_model, RF_model, KNN_model, DT_model, SVM_model = train_all_models(embeddings_path,
                                                                  screen_number,
                                                                  scrambled)

    embeddings_path = '../data/embedding_dataframes_clean/unirep64_preconcat_linker_scrambled_final_embeddings.csv'
    screen_number_scram = 1
    scrambled_scram = True

    lasso_model_scram, RF_model_scram, KNN_model_scram, DT_model_scram, SVM_model_scram = train_all_models(embeddings_path,
                                                                  screen_number,
                                                                  scrambled_scram)


    embeddings_path = '../data/embedding_dataframes_clean/unirep64_preconcat_linker_final_embeddings.csv'
    embeddings_scrambled_path = '../data/embedding_dataframes_clean/unirep64_preconcat_linker_scrambled_final_embeddings.csv'
    ntrials = 100

    embeddings_df = pd.read_csv(embeddings_path, index_col=0)
    embeddings_scrambed_df = pd.read_csv(embeddings_scrambled_path, index_col=0)

    # split training features and target values
    unirep64_feats = 64 # excluding physiochemical features
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

        r2_lasso, r2_RF, r2_NN, r2_DT,r2_SVM, r2_lasso_scram, r2_RF_scram, r2_NN_scram, r2_DT_scram, r2_SVM_scram = scramble_predictions(X_train, X_test, y_train, y_test,
                            X_scrambled_train, X_scrambled_test,
                            y_scrambled_train, y_scrambled_test,
                            lasso_model, RF_model,KNN_model, DT_model, SVM_model)

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
    results_df.to_csv('../data/machine_learning/regression_preconcat_linker_scramble_comparison.csv')

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
                         "SVM_model_stats": DT_model_stats}
    results_stats_df = pd.DataFrame(results_stats_dict)
    results_stats_df.to_csv('../data/machine_learning/regression_preconcat_linker_scramble_comparison_stats.csv')

    print('loading data')
    embeddings_path = '../data/embedding_dataframes_clean/unirep64_preconcat_linker_final_embeddings.csv'
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)
    embeddings_df = embeddings_df.iloc[:, np.r_[:64, -1]]
    unirep64_feats = 64

    regulator_combos = list(embeddings_df.index)
    individual_regulators_list = []
    for reg_combo in regulator_combos:
        split_combos = reg_combo.split('_')
        reg1 = split_combos[0]
        reg2 = split_combos[1]
        individual_regulators_list.append(reg1)
        individual_regulators_list.append(reg2)
    individual_regulators_unique_list = list(set(individual_regulators_list))

    generalizability_results = pd.DataFrame(index=individual_regulators_unique_list)
    lasso_results_pearson = []
    lasso_results_spearman = []
    lasso_results_Rsquared = []

    RF_results_pearson = []
    RF_results_spearman = []
    RF_results_Rsquared = []

    KNN_results_pearson = []
    KNN_results_spearman = []
    KNN_results_Rsquared = []

    DT_results_pearson = []
    DT_results_spearman = []
    DT_results_Rsquared = []

    lasso_results_pearson_scram = []
    lasso_results_spearman_scram = []
    lasso_results_Rsquared_scram = []

    RF_results_pearson_scram = []
    RF_results_spearman_scram = []
    RF_results_Rsquared_scram = []

    KNN_results_pearson_scram = []
    KNN_results_spearman_scram = []
    KNN_results_Rsquared_scram = []

    DT_results_pearson_scram = []
    DT_results_spearman_scram = []
    DT_results_Rsquared_scram = []

    print('testing models')
    for protein in individual_regulators_unique_list:
        print(f'current protein: {protein}')

        protein_start = protein + '_'
        protein_end = '_' + protein
        embeddings_training_temp_df = embeddings_df.loc[(~embeddings_df.index.str.contains(protein_start)) & (~embeddings_df.index.str.contains(protein_end))]
        embeddings_testing_temp_df = embeddings_df.loc[(embeddings_df.index.str.contains(protein_start)) | (embeddings_df.index.str.contains(protein_end))]

        X_train = embeddings_training_temp_df.iloc[:,0:unirep64_feats].dropna(axis=1)
        y_train = embeddings_training_temp_df.copy().iloc[:, [-1]]

        X_test = embeddings_testing_temp_df.iloc[:,0:unirep64_feats].dropna(axis=1)
        y_test = embeddings_testing_temp_df.copy().iloc[:, [-1]]

        X_scrambled_df = embeddings_scrambed_df.iloc[:,0:unirep64_feats].dropna(axis=1)#
        y_scrambled_df = embeddings_scrambed_df.copy().iloc[:, [-1]]
        X_scrambled_train = pd.concat([X_train, X_scrambled_df], axis=1).dropna().iloc[:,unirep64_feats:]
        X_scrambled_test = pd.concat([X_test, X_scrambled_df], axis=1).dropna().iloc[:,unirep64_feats:]
        y_scrambled_train = pd.concat([y_train, y_scrambled_df], axis=1).dropna().iloc[:,1:]
        y_scrambled_test = pd.concat([y_test, y_scrambled_df], axis=1).dropna().iloc[:,1:]


        # lasso model
        lasso_model.fit(X_train, y_train)
        y_pred_lasso = lasso_model.predict(X_test)
        lasso_pearsonr2 = stats.pearsonr(y_test['flow_seq_score'], y_pred_lasso)[0]**2
        lasso_spearman = stats.spearmanr(y_test['flow_seq_score'], y_pred_lasso)[0]
        lasso_rsquared = r2_score(y_test['flow_seq_score'], y_pred_lasso)

        # lasso test of scram
        y_pred_lasso_scram = lasso_model.predict(X_scrambled_test)
        lasso_pearsonr2_scram = stats.pearsonr(y_scrambled_test['flow_seq_score'], y_pred_lasso_scram)[0]**2
        lasso_spearman_scram = stats.spearmanr(y_scrambled_test['flow_seq_score'], y_pred_lasso_scram)[0]
        lasso_rsquared_scram = r2_score(y_scrambled_test['flow_seq_score'], y_pred_lasso_scram)

        lasso_results_pearson.append(lasso_pearsonr2)
        lasso_results_spearman.append(lasso_spearman)
        lasso_results_Rsquared.append(lasso_rsquared)

        lasso_results_pearson_scram.append(lasso_pearsonr2_scram)
        lasso_results_spearman_scram.append(lasso_spearman_scram)
        lasso_results_Rsquared_scram.append(lasso_rsquared_scram)

        # RF model
        RF_model.fit(X_train, y_train.values.ravel())
        y_pred_RF = RF_model.predict(X_test)
        RF_pearsonr2 = stats.pearsonr(y_test['flow_seq_score'], y_pred_RF)[0]**2
        RF_spearman = stats.spearmanr(y_test['flow_seq_score'], y_pred_RF)[0]
        RF_rsquared = r2_score(y_test['flow_seq_score'], y_pred_RF)

        # RF model_scram
        y_pred_RF_scram = RF_model.predict(X_scrambled_test)
        RF_pearsonr2_scram = stats.pearsonr(y_scrambled_test['flow_seq_score'], y_pred_RF_scram)[0]**2
        RF_spearman_scram = stats.spearmanr(y_scrambled_test['flow_seq_score'], y_pred_RF_scram)[0]
        RF_rsquared_scram = r2_score(y_scrambled_test['flow_seq_score'], y_pred_RF_scram)

        RF_results_pearson.append(RF_pearsonr2)
        RF_results_spearman.append(RF_spearman)
        RF_results_Rsquared.append(RF_rsquared)

        RF_results_pearson_scram.append(RF_pearsonr2_scram)
        RF_results_spearman_scram.append(RF_spearman_scram)
        RF_results_Rsquared_scram.append(RF_rsquared_scram)

        # KNN model
        KNN_model.fit(X_train, y_train.values.ravel())
        y_pred_NN = KNN_model.predict(X_test)
        NN_pearsonr2 = stats.pearsonr(y_test['flow_seq_score'], y_pred_NN)[0]**2
        NN_spearman = stats.spearmanr(y_test['flow_seq_score'], y_pred_NN)[0]
        NN_rsquared = r2_score(y_test['flow_seq_score'], y_pred_NN)

        # KNN model_scram
        y_pred_NN_scram = KNN_model.predict(X_scrambled_test)
        NN_pearsonr2_scram = stats.pearsonr(y_scrambled_test['flow_seq_score'], y_pred_NN_scram)[0]**2
        NN_spearman_scram = stats.spearmanr(y_scrambled_test['flow_seq_score'], y_pred_NN_scram)[0]
        NN_rsquared_scram = r2_score(y_scrambled_test['flow_seq_score'], y_pred_NN_scram)

        KNN_results_pearson.append(NN_pearsonr2)
        KNN_results_spearman.append(NN_spearman)
        KNN_results_Rsquared.append(NN_rsquared)

        KNN_results_pearson_scram.append(NN_pearsonr2_scram)
        KNN_results_spearman_scram.append(NN_spearman_scram)
        KNN_results_Rsquared_scram.append(NN_rsquared_scram)

        # DT model
        DT_model.fit(X_train, y_train.values.ravel())
        y_pred_DT = DT_model.predict(X_test)
        DT_pearsonr2 = stats.pearsonr(y_test['flow_seq_score'], y_pred_DT)[0]**2
        DT_spearman = stats.spearmanr(y_test['flow_seq_score'], y_pred_DT)[0]
        DT_rsquared = r2_score(y_test['flow_seq_score'], y_pred_DT)

        # DT model_scram
        y_pred_DT_scram = DT_model.predict(X_scrambled_test)
        DT_pearsonr2_scram = stats.pearsonr(y_scrambled_test['flow_seq_score'], y_pred_DT_scram)[0]**2
        DT_spearman_scram = stats.spearmanr(y_scrambled_test['flow_seq_score'], y_pred_DT_scram)[0]
        DT_rsquared_scram = r2_score(y_scrambled_test['flow_seq_score'], y_pred_DT_scram)

        DT_results_pearson.append(DT_pearsonr2)
        DT_results_spearman.append(DT_spearman)
        DT_results_Rsquared.append(DT_rsquared)

        DT_results_pearson_scram.append(DT_pearsonr2_scram)
        DT_results_spearman_scram.append(DT_spearman_scram)
        DT_results_Rsquared_scram.append(DT_rsquared_scram)

    generalizability_results['lasso_pearson_r2'] = lasso_results_pearson
    generalizability_results['lasso_spearman'] = lasso_results_spearman
    generalizability_results['lasso_rsquared'] = lasso_results_Rsquared

    generalizability_results['RF_pearson_r2'] = RF_results_pearson
    generalizability_results['RF_spearman'] = RF_results_spearman
    generalizability_results['RF_rsquared'] = RF_results_Rsquared

    generalizability_results['KNN_pearson_r2'] = KNN_results_pearson
    generalizability_results['KNN_spearman'] = KNN_results_spearman
    generalizability_results['KNN_rsquared'] = KNN_results_Rsquared

    generalizability_results['DT_pearson_r2'] = DT_results_pearson
    generalizability_results['DT_spearman'] = DT_results_spearman
    generalizability_results['DT_rsquared'] = DT_results_Rsquared

    generalizability_results['lasso_pearson_r2_scram'] = lasso_results_pearson_scram
    generalizability_results['lasso_spearman_scram'] = lasso_results_spearman_scram
    generalizability_results['lasso_rsquared_scram'] = lasso_results_Rsquared_scram

    generalizability_results['RF_pearson_r2_scram'] = RF_results_pearson_scram
    generalizability_results['RF_spearman_scram'] = RF_results_spearman_scram
    generalizability_results['RF_rsquared_scram'] = RF_results_Rsquared_scram

    generalizability_results['KNN_pearson_r2_scram'] = KNN_results_pearson_scram
    generalizability_results['KNN_spearman_scram'] = KNN_results_spearman_scram
    generalizability_results['KNN_rsquared_scram'] = KNN_results_Rsquared_scram

    generalizability_results['DT_pearson_r2_scram'] = DT_results_pearson
    generalizability_results['DT_spearman_scram'] = DT_results_spearman
    generalizability_results['DT_rsquared_scram'] = DT_results_Rsquared

    generalizability_results.to_csv('../data/machine_learning/leave_one_protein_out_test_preconcat_linker.csv')

    print('loading data')
    embeddings_path = '../data/embedding_dataframes_clean/unirep64_preconcat_linker_scrambled_final_embeddings.csv'
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)
    embeddings_df = embeddings_df.iloc[:, np.r_[:64, -1]]
    unirep64_feats = 64

    # these are actually true embeddings (just repeating code)
    embeddings_scrambed_path = '../data/embedding_dataframes_clean/unirep64_preconcat_linker_final_embeddings.csv'
    embeddings_scrambed_df = pd.read_csv(embeddings_scrambed_path, index_col=0)

    regulator_combos = list(embeddings_df.index)
    individual_regulators_list = []
    for reg_combo in regulator_combos:
        split_combos = reg_combo.split('_')
        reg1 = split_combos[0]
        reg2 = split_combos[1]
        individual_regulators_list.append(reg1)
        individual_regulators_list.append(reg2)
    individual_regulators_unique_list = list(set(individual_regulators_list))

    generalizability_results = pd.DataFrame(index=individual_regulators_unique_list)
    lasso_results_pearson = []
    lasso_results_spearman = []
    lasso_results_Rsquared = []

    RF_results_pearson = []
    RF_results_spearman = []
    RF_results_Rsquared = []

    KNN_results_pearson = []
    KNN_results_spearman = []
    KNN_results_Rsquared = []

    DT_results_pearson = []
    DT_results_spearman = []
    DT_results_Rsquared = []

    # these will be for train on scramble, test on true
    lasso_results_pearson_scram = []
    lasso_results_spearman_scram = []
    lasso_results_Rsquared_scram = []

    RF_results_pearson_scram = []
    RF_results_spearman_scram = []
    RF_results_Rsquared_scram = []

    KNN_results_pearson_scram = []
    KNN_results_spearman_scram = []
    KNN_results_Rsquared_scram = []

    DT_results_pearson_scram = []
    DT_results_spearman_scram = []
    DT_results_Rsquared_scram = []

    print('testing models')
    for protein in individual_regulators_unique_list:
        print(f'current protein: {protein}')

        protein_start = protein + '_'
        protein_end = '_' + protein
        embeddings_training_temp_df = embeddings_df.loc[(~embeddings_df.index.str.contains(protein_start)) & (~embeddings_df.index.str.contains(protein_end))]
        embeddings_testing_temp_df = embeddings_df.loc[(embeddings_df.index.str.contains(protein_start)) | (embeddings_df.index.str.contains(protein_end))]

        X_train = embeddings_training_temp_df.iloc[:,0:unirep64_feats].dropna(axis=1)
        y_train = embeddings_training_temp_df.copy().iloc[:, [-1]]

        X_test = embeddings_testing_temp_df.iloc[:,0:unirep64_feats].dropna(axis=1)
        y_test = embeddings_testing_temp_df.copy().iloc[:, [-1]]

        # again, these are actually the true sequences
        X_scrambled_df = embeddings_scrambed_df.iloc[:,0:unirep64_feats].dropna(axis=1)#
        y_scrambled_df = embeddings_scrambed_df.copy().iloc[:, [-1]]
        X_scrambled_train = pd.concat([X_train, X_scrambled_df], axis=1).dropna().iloc[:,unirep64_feats:]
        X_scrambled_test = pd.concat([X_test, X_scrambled_df], axis=1).dropna().iloc[:,unirep64_feats:]
        y_scrambled_train = pd.concat([y_train, y_scrambled_df], axis=1).dropna().iloc[:,1:]
        y_scrambled_test = pd.concat([y_test, y_scrambled_df], axis=1).dropna().iloc[:,1:]

        # lasso model
        lasso_model.fit(X_train, y_train)
        y_pred_lasso = lasso_model.predict(X_test)
        lasso_pearsonr2 = stats.pearsonr(y_test['flow_seq_score'], y_pred_lasso)[0]**2
        lasso_spearman = stats.spearmanr(y_test['flow_seq_score'], y_pred_lasso)[0]
        lasso_rsquared = r2_score(y_test['flow_seq_score'], y_pred_lasso)

        lasso_results_pearson.append(lasso_pearsonr2)
        lasso_results_spearman.append(lasso_spearman)
        lasso_results_Rsquared.append(lasso_rsquared)

        # RF model
        RF_model.fit(X_train, y_train.values.ravel())
        y_pred_RF = RF_model.predict(X_test)
        RF_pearsonr2 = stats.pearsonr(y_test['flow_seq_score'], y_pred_RF)[0]**2
        RF_spearman = stats.spearmanr(y_test['flow_seq_score'], y_pred_RF)[0]
        RF_rsquared = r2_score(y_test['flow_seq_score'], y_pred_RF)

        # RF model_scram
        y_pred_RF_scram = RF_model.predict(X_scrambled_test)
        RF_pearsonr2_scram = stats.pearsonr(y_scrambled_test['flow_seq_score'], y_pred_RF_scram)[0]**2
        RF_spearman_scram = stats.spearmanr(y_scrambled_test['flow_seq_score'], y_pred_RF_scram)[0]
        RF_rsquared_scram = r2_score(y_scrambled_test['flow_seq_score'], y_pred_RF_scram)

        RF_results_pearson.append(RF_pearsonr2)
        RF_results_spearman.append(RF_spearman)
        RF_results_Rsquared.append(RF_rsquared)

        RF_results_pearson_scram.append(RF_pearsonr2_scram)
        RF_results_spearman_scram.append(RF_spearman_scram)
        RF_results_Rsquared_scram.append(RF_rsquared_scram)

        # KNN model
        KNN_model.fit(X_train, y_train.values.ravel())
        y_pred_NN = KNN_model.predict(X_test)
        NN_pearsonr2 = stats.pearsonr(y_test['flow_seq_score'], y_pred_NN)[0]**2
        NN_spearman = stats.spearmanr(y_test['flow_seq_score'], y_pred_NN)[0]
        NN_rsquared = r2_score(y_test['flow_seq_score'], y_pred_NN)

        KNN_results_pearson.append(NN_pearsonr2)
        KNN_results_spearman.append(NN_spearman)
        KNN_results_Rsquared.append(NN_rsquared)

        # DT model
        DT_model.fit(X_train, y_train.values.ravel())
        y_pred_DT = DT_model.predict(X_test)
        DT_pearsonr2 = stats.pearsonr(y_test['flow_seq_score'], y_pred_DT)[0]**2
        DT_spearman = stats.spearmanr(y_test['flow_seq_score'], y_pred_DT)[0]
        DT_rsquared = r2_score(y_test['flow_seq_score'], y_pred_DT)

        DT_results_pearson.append(DT_pearsonr2)
        DT_results_spearman.append(DT_spearman)
        DT_results_Rsquared.append(DT_rsquared)

    generalizability_results['lasso_pearson_r2'] = lasso_results_pearson
    generalizability_results['lasso_spearman'] = lasso_results_spearman
    generalizability_results['lasso_rsquared'] = lasso_results_Rsquared

    generalizability_results['RF_pearson_r2'] = RF_results_pearson
    generalizability_results['RF_spearman'] = RF_results_spearman
    generalizability_results['RF_rsquared'] = RF_results_Rsquared

    generalizability_results['KNN_pearson_r2'] = KNN_results_pearson
    generalizability_results['KNN_spearman'] = KNN_results_spearman
    generalizability_results['KNN_rsquared'] = KNN_results_Rsquared

    generalizability_results['DT_pearson_r2'] = DT_results_pearson
    generalizability_results['DT_spearman'] = DT_results_spearman
    generalizability_results['DT_rsquared'] = DT_results_Rsquared

    # trained on scrambled; tested on true

    generalizability_results['RF_pearson_r2_scram'] = RF_results_pearson_scram
    generalizability_results['RF_spearman_scram'] = RF_results_spearman_scram
    generalizability_results['RF_rsquared_scram'] = RF_results_Rsquared_scram
    
    generalizability_results.to_csv('../data/machine_learning/leave_one_protein_out_test_preconcat_linker_scrambled.csv')


if __name__ == "__main__":
    main()
