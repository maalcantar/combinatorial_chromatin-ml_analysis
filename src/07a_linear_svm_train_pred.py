#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 18 11:29:24 2022
@author: alcantar
example run: python 07a_linear_svm_train_pred.py -i ../data/embedding_dataframes_clean/unirep64_final_embeddings.csv -c 0.6 -n 100
"""
# activate virtual enviroment before running script
# source activate combicr_ml

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import sys
import warnings
import argparse
import os
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

import argparse

def one_hot_encode_labels(flow_seq_score, cutoff=0.6):

    '''
    convert flow-seq scores into binary values. specifically,
    if the flow-seq score is >0.6, it is considered a strong activator
    (cutoff is based on validations) and assigned a value of 1. otherwise,
    it is assigned a value of 0.

    PARAMETERS
    --------------------
    flow_seq_score: list or pandas dataframe row
        flow-seq scores
    cutoff: float
        flow seq score cutoff for assigning labels

    RETURNS
    --------------------
    one_hot_encoded_labels: list of integers
        one hot encoded flow seq scores
    '''

    one_hot_encoded_labels =[0 if label < cutoff else 1 for label in flow_seq_score]

    return(one_hot_encoded_labels)

def to_array(str_array):
    float_array = [float(string) for string in str_array]
    return(float_array)

def plotROC(ensemble_df, output_path, cutoff, shuffled, save=False):

    '''
    plot the mean ROC curve and include confidence interval

    PARAMETERS
    --------------------
    ensemble_df: pandas dataframe
        dataframe with ensemble statistics for ROC and PRC curves
    output_path: string
        output path for figures
    cutoff: float
        cutoff used to assign labels (for naming output figure)
    shuffled: boolean
        indicate whether data labels have been shuffled (for output name)
    save: boolean
        indicate whether to save output image

    RETURNS
    --------------------
    none

    '''
    # define points at which to interpolate tpr vs. fpr
    fpr_mean = np.linspace(0, 1, 500)
    interp_tprs = []
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(1, 1, 1)

    # loop through ensemble results
    for _, row in ensemble_df.iterrows():
        fpr = to_array(row['fpr'])
        tpr = to_array(row['tpr'])

        # interpolate tpr vs. fpr
        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    # calculate summary statistics for ROC curve
    tpr_mean = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std = 2*np.std(interp_tprs, axis=0)
    auc = np.mean(ensemble_df['roc_auc'])
    auc_std = 2*np.std(ensemble_df['roc_auc'])

    # plot mean TPR vs FPR +/- 2 standard deviations
    ax.plot(fpr_mean, tpr_mean,
                label= f'auROC = {str(round(auc,3))} +/- {str(round(auc_std,3))}')
    ax.fill_between(fpr_mean, (tpr_mean-tpr_std), (tpr_mean+tpr_std), color='b', alpha=.1)

    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle=':')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")

    if save:
        if shuffled:
            output_name = output_path + 'ROC_' + str(cutoff).replace('.','-') + '_cutoff_shuffled'
        else:
            output_name = output_path + 'ROC_' + str(cutoff).replace('.','-') + '_cutoff'

        plt.savefig(output_name+'.png')
        plt.savefig(output_name+'.pdf')
        plt.savefig(output_name+'.svg')

def plotPRC(summary_df, ensemble_df, output_path, cutoff, shuffled, save=False):

    '''
    plot the median PRC curve and include confidence interval in plot legend.

    PARAMETERS
    --------------------
    summary_df: pandas dataframe
        dataframe with summary statistics for ROC and PRC curves
    ensemble_df: pandas dataframe
        dataframe with all statistics for ROC and PRC
    output_path: string
        output path for figures
    cutoff: float
         cutoff used to assign labels (for naming output figure)
     shuffled: boolean
         indicate whether data labels have been shuffled (for output name)
     save: boolean
       indicate whether to save output image

    RETURNS
    --------------------
    none

    '''

    fig = plt.figure(dpi=400,)
    ax = fig.add_subplot(1, 1, 1)

    for ind, row in summary_df.iterrows():
        precision = to_array(row['precision'])
        recall = to_array(row['recall'])

    # calculate statistics for average prc
    precision_mean = np.mean(precision, axis=0)
    precision_std = 2*np.std(precision, axis=0)
    ap_mean = np.mean(ensemble_df['AP'])
    ap_std = 2*np.std(ensemble_df['AP'])

    ax.step(recall, precision, where='post',
           label=f'AP={str(round(ap_mean,3))} +/- {str(round(ap_std,3))}')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    plt.title('PRC')
    plt.legend(loc='lower left')

    if save:
        if shuffled:
            output_name = output_path + 'PRC_' + str(cutoff).replace('.','-') + '_cutoff_shuffled'
        else:
            output_name = output_path + 'PRC_' + str(cutoff).replace('.','-') + '_cutoff'

        plt.savefig(output_name+'.png')
        plt.savefig(output_name+'.pdf')
        plt.savefig(output_name+'.svg')

def train_SVM(X_train, y_train,
          kernel='linear', alpha=0.001, threshold=0.01,
          save=False, pickle_file='model.pkl', results_file='results.csv'):

    '''
    train a support vector machine using grid search and cross validation
    to identify the best model.

    PARAMETERS
    --------------------
    X_train: pandas dataframe
        dataframe with features
    y_train: pandas dataframe
        dataframe with labels
    kernel: string
        svm kernel
    alpha: float
        L1 regularization parameter
    threshold: float
        weight threshold to maintain in model
    save: boolean
        indicate whether to save model
    pickle_file: string
        output path for model in pickle format
    results_file: string
        output path for training results file

    RETURNS
    --------------------
    best_model
        best svm model
    '''

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=4, random_state=42)
    if kernel == 'linear':
        # hinge loss for linear SVM; L1 for sparsity; don't shuffle data each epoch (reproducibility);
        # balanced = don't bias classes (class imbalance)
        clf = linear_model.SGDClassifier(loss='hinge', penalty='l1', shuffle=False,
                                         class_weight='balanced' , alpha=alpha)

        featSelection = SelectFromModel(clf, threshold=threshold) # threshold for feature selection

        # pipeline for SVM model -- this is still for cross-validation
        model = Pipeline([
                          ('fs', featSelection),
                          ('clf', clf),
                        ])

        # range of threshold values to test
        param_grid = {'fs__threshold': np.linspace(0.001, 10, num=100),
                      'clf__alpha': np.logspace(-4, -2, num=2)}
    else:
        # setting up non-linear model (either radial basis function or sigmoid)
        model = svm.SVC(kernel=kernel, C = 10.0, gamma=0.1, cache_size=500)
        # setting up grid for hyperparameter search
        param_grid = {'C': np.logspace(-4, 3, num=7),
                      'gamma': np.logspace(-5, 2, num=7)}
    # performing grid search
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, error_score=0.0)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_CVscore = grid.best_score_

    print("The best parameters are %s with a score of %0.2f"% (best_params, best_CVscore))
    best_model = grid.best_estimator_

    if save:
        with open(pickle_file, 'wb') as handle:
            pickle.dump(best_model, handle)

        if kernel == 'linear':
            CVresults = pd.DataFrame(data=grid.cv_results_,
                                     columns=['param_fs__threshold',
                                              'param_clf__alpha',
                                              'mean_test_score'])
        else:
            CVresults = pd.DataFrame(data=grid.cv_results_,
                                     columns=['param_C', 'param_gamma',
                                              'mean_test_score'])
    CVresults.to_csv(results_file)

    return(best_model)


def summary_stats(ens):
    ens = ens.dropna(axis=0)
    cols_to_med = ['roc_auc', 'AP', 'accuracy']

    summary = pd.DataFrame()

    for col in cols_to_med:
        vals = ens[col].values
        med_idx = np.argsort(vals)[len(vals)//2]
        median = vals[med_idx]
        ci = np.percentile(vals, (2.5, 97.5))

        if np.isnan(median): #if nan
            continue

        summary[col+'_ci'] = [ci]
        summary[col+'_median'] = median


        med_row = ens.iloc[med_idx, :]
        if (col == 'roc_auc'):
            fpr = med_row['fpr']
            tpr = med_row['tpr']
            temp = pd.DataFrame({'fpr': [fpr], 'tpr': [tpr]})
            summary = pd.merge(summary, temp, how='left',
                               left_index=True, right_index=True)
        if (col == 'AP'):
            precision = med_row['precision']
            recall = med_row['recall']
            temp = pd.DataFrame({'precision': [precision],
                                 'recall': [recall]})
            summary = pd.merge(summary, temp, how='left',
                               left_index=True, right_index=True)
    return(summary)

def svm_trials(X_df, y_df, model, cutoff=0.6, ntrials=100, shuffle=False):

    '''
    train and predict optimal svm for multiple train-test splits in
    order to obtain confidence intervals. option to shuffle y labels
    is also available.

    PARAMETERS
    --------------------
    X_df: pandas dataframe
        dataframe with protein embeddings
    y_df: pandas dataframe
        dataframe with y-labels
    model: model
        svm model with optimal hyperparameters
    ntrials: int
        number of trials to run
    shuffle: boolean
        indicate whether to shuffle labels

    RETURNS
    --------------------
    all_results_ensemble_df: pandas dataframe
        dataframe with prediction results for all trials


    '''

    all_results_ensemble = []
    all_weights_ens = []
    number_trials = ntrials


    if shuffle:
        print('shuffled data')
        if cutoff==0.6:
            y_df = y_df.sample(frac=1, random_state=1).reset_index(drop=True)
        elif cutoff == 0.5:
            y_df = y_df.sample(frac=1, random_state=2).reset_index(drop=True)
        else:
            y_df = y_df.sample(frac=1).reset_index(drop=True)
    else:
        print('non-shuffled data')

    successful_trials = 0
    trial=0
    # for trial in range(number_trials):
    while successful_trials < number_trials:
        try:
            if successful_trials % 10 == 0:
                print(f'current trial: {successful_trials}')

            test_size=0.20
            X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                                test_size=test_size,
                                                                random_state=trial)
            model.fit(X_train, y_train['flow_seq_score'])

            try: #for SVM
                y_score = model.predict(X_test)
                y_score = model.decision_function(X_test)
            except AttributeError: #for RF
                y_score = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            averagePrecision = average_precision_score(y_test, y_score)
            precision, recall, _ = precision_recall_curve(y_test, y_score)

            modelAccuracy = model.score(X_test, y_test)
            results_df = pd.DataFrame({'fpr': [fpr], 'tpr': [tpr],
                                       'roc_auc': roc_auc,
                                       'precision': [precision],
                                       'recall': [recall],
                                       'AP': averagePrecision,
                                       'accuracy': modelAccuracy,
                                      'seed':trial})
            all_results_ensemble.append(results_df)

            successful_trials+=1
            trial+=1

            if not shuffle:
                all_weights = feat_SVM(X_train, y_train, model)

                if all_weights is not None:
                    all_weights['seed'] = trial
                    all_weights_ens.append(all_weights)

        except ValueError:
            print('unsuccesful trial. trying new seed')
            trial+=1

    all_results_ensemble_df = pd.concat(all_results_ensemble, sort=False).set_index('seed')

    output_weight_path = '../data/machine_learning/linear_svm/weights_' + str(cutoff).replace('.','-') +'cutoff.csv'
    if not shuffle:
        if all_weights_ens:
            all_weights_ens = pd.concat(all_weights_ens, sort=False).set_index('seed')
            #Normalize weights per row to get relative importances
            all_weights_ens = get_weights(all_weights_ens)
            all_weights_ens.to_csv(output_weight_path)

    return(all_results_ensemble_df)

def feat_SVM(X_train, y_train, model):
    model.fit(X_train, y_train.values.ravel())

    try: #for SVM
        weights = pd.DataFrame([model.named_steps["clf"].coef_[0]],
                                columns=X_train.columns[model.named_steps["fs"].get_support()])
    except AttributeError: #for RF
        try:
            weights = pd.DataFrame([model.feature_importances_],
                                   columns=X_train.columns)
        except AttributeError:
            weights = None
    return weights

def get_weights(weights_ens):
    weights_ens = weights_ens.abs()
    weights_ens = weights_ens.div(weights_ens.sum(axis=1), axis=0)
    weights_ens = weights_ens.sum(axis=0).transpose()
    weights_ens = pd.DataFrame(data=weights_ens,
                               columns=['Feature importance'])
    #weights_ens = weights_to_pathways(weights_ens, chemfile)
    weights_ens = weights_ens.sort_values(by='Feature importance',
                                          ascending=False)
    return weights_ens


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to embeddings dataframe')
    parser.add_argument('-c', help='flow-seq score cutoff')
    parser.add_argument('-n', help='number of svm trials to run')


    args = parser.parse_args()
    embeddings_path = args.i
    cutoff = float(args.c)
    ntrials = int(args.n)

    unirep_embeddings_path = embeddings_path #'../../data/embedding_dataframes_clean/unirep64_final_embeddings.csv'
    unirep_embeddings_df = pd.read_csv(unirep_embeddings_path, index_col=0)
    unirep64_feats = 128 # excluding physiochemical features

    X_df = unirep_embeddings_df.iloc[:,0:unirep64_feats].dropna(axis=1)#
    y_df = unirep_embeddings_df.copy().iloc[:, [-1]]
    y_list = list(y_df['flow_seq_score'].copy())
    y_df['flow_seq_score'] = one_hot_encode_labels(y_list, cutoff=cutoff)#one_hot_encode_labels(y_df.copy()['flow_seq_score'])
    print('values one hot encoded')

    test_size = 0.20
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                        test_size=test_size,
                                                        random_state=random_state)
    pickle_output_ml_path = '../data/machine_learning/'
    # create ouput folders if needed
    CHECK_FOLDER_MAIN_ML = os.path.isdir(pickle_output_ml_path)
    if not CHECK_FOLDER_MAIN_ML:
        os.makedirs(pickle_output_ml_path)
        print(f"created folder : {pickle_output_ml_path}")

    pickle_output_ml_svm_path = pickle_output_ml_path+ 'linear_svm/'
    # create ouput folders if needed
    CHECK_FOLDER_MAIN_ML_SVM = os.path.isdir(pickle_output_ml_svm_path)
    if not CHECK_FOLDER_MAIN_ML_SVM:
        os.makedirs(pickle_output_ml_svm_path)
        print(f"created folder : {pickle_output_ml_svm_path}")

    print('training model')
    kernel = 'linear'
    model_output_path = pickle_output_ml_path + 'linear_svm/linear_SVM_' + str(cutoff).replace('.','-') + '_cutoff.pkl'
    result_output_path= pickle_output_ml_path + 'linear_svm/CVresults_' + str(cutoff).replace('.','-') + '_cutoff.csv'
    model = train_SVM(X_train, y_train['flow_seq_score'], kernel=kernel, save=True,
                  pickle_file=model_output_path, results_file=result_output_path)

    filename_dir='../figs/machine_learning/linear_SVM/'
    # regular trial
    print(ntrials)

    print('running trials')
    all_results_ensemble_df = svm_trials(X_df, y_df, model, cutoff, ntrials=ntrials, shuffle=False)
    all_results_sum = summary_stats(all_results_ensemble_df)
    shuffled=False
    plotROC(all_results_ensemble_df,filename_dir, cutoff, shuffled, save=True)
    plotPRC(all_results_sum, all_results_ensemble_df, filename_dir, cutoff, shuffled, save=True)

    # shuffled trial
    all_results_ensemble_shuffled_df = svm_trials(X_df, y_df, model, cutoff, ntrials=ntrials, shuffle=True)

    if all_results_ensemble_shuffled_df is not None:
        all_results_shuffled_sum = summary_stats(all_results_ensemble_shuffled_df)
        shuffled=True
        plotROC(all_results_ensemble_shuffled_df,filename_dir, cutoff, shuffled, save=True)
        plotPRC(all_results_shuffled_sum, all_results_ensemble_shuffled_df, filename_dir, cutoff, shuffled, save=True)


if __name__ == "__main__":
    main()
