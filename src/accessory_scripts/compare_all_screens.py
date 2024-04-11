#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on March 1 18:29:07 2024
@author: alcantar
example run: python compare_all_screens.py
"""
# activate virtual enviroment before running script
# source activate combicr_ml

import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

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

def read_preprocess_df(screen_no,
                       screen_path):

    '''
    creates a minmal dataframe with the flow seq scores obtained for a given screen.
    this function also outputs the indices of the creates dataframe to simplify
    subsequent steps in this script

    PARAMETERS
    --------------------
    screen_no: int
        current screen number [1,2,3,4]
    screen_path: str
        path to csv file with embeddings and flow-seq score

    RETURNS
    --------------------
    screen_no_str: str
        screen number as a string. this helps with creating the updated dataframe
    screen_combos_tmp: list of str
        combos obtained from current screen
    screen_df_tmp: pandas dataframe
        minimal pandas dataframe with the flow-seq score
    '''

    screen_no_str = "screen_" + str(screen_no)
    column_name_tmp = "flow_seq_score_" + screen_no_str
    screen_df_tmp = pd.read_csv(screen_path, index_col=0)[["flow_seq_score"]].rename(columns={"flow_seq_score": column_name_tmp})
    screen_combos_tmp = list(screen_df_tmp.index)

    return(screen_no_str, screen_combos_tmp, screen_df_tmp)

def main():
    # paths to embeddings and flow-seq scores
    make_dir('../../figs/screen_comparisons/') # create output folder for figures
    screen1_results_path = '../../data/embedding_dataframes_clean/unirep64_final_embeddings.csv'
    screen2_results_path = '../../data/embedding_dataframes_screen2_clean/unirep64_final_screen2_embeddings.csv'
    screen3_results_path = '../../data/embedding_dataframes_screen3_clean/unirep64_final_screen3_embeddings.csv'
    screen4_results_path = '../../data/embedding_dataframes_screen4_clean/unirep64_final_screen4_embeddings.csv'

    screens_all_paths = [screen1_results_path,
                        screen2_results_path,
                        screen3_results_path,
                        screen4_results_path]

    screen_df_dict = dict()
    screens_all_combos = []
    for screen_no, screen_path in enumerate(screens_all_paths,1):
        screen, combos, df = read_preprocess_df(screen_no,
                                                screen_path)
        screens_all_combos.append(combos)
        screen_df_dict.update({screen:df})

    shared_combos = set.intersection(*map(set,screens_all_combos))
    num_shared_combos = len(shared_combos)
    print(f'There are {str(num_shared_combos)} shared combos across the four screens')

    screen_df_final_dict = dict()
    for screen_no, screen_df_tmp in enumerate(screen_df_dict.values(),1):
        screen_final_df_tmp = screen_df_tmp.copy()[screen_df_tmp.index.isin(shared_combos)]
        screen_df_final_dict.update({f"screen_{str(screen_no)}": screen_final_df_tmp})
    screen_all_df = screen_df_final_dict['screen_1'].join(screen_df_final_dict['screen_2'])
    screen_all_df = screen_all_df.join(screen_df_final_dict['screen_3'])
    screen_all_df = screen_all_df.join(screen_df_final_dict['screen_4'])
    screen_all_df.to_csv('../../data/all_screen_data.csv') # output to csv for replotting


    # plot all comparisons (including redundant screen combinations)
    # these plots will be remade in PRISM / GRAPHPAD
    for screen_i in range(1,5):
        for screen_j in range(1,5):
            plt.rcParams.update({'font.size': 18,
                                 'font.family':'arial'})
            print(f'comparing screen{str(screen_i)} and screen{str(screen_j)}')
            screen_1_name = 'flow_seq_score_screen_' + str(screen_i)
            screen_2_name = 'flow_seq_score_screen_' + str(screen_j)
            x_data = list(screen_all_df[screen_1_name])
            y_data = list(screen_all_df[screen_2_name])
            spearmanR = round(stats.spearmanr(x_data, y_data)[0],3)
            pearsonR = round(stats.pearsonr(x_data, y_data)[0]**2,3)
            fig_train = plt.figure(dpi=400, figsize=(8,8))
            ax = fig_train.add_subplot(1, 1, 1)
            ax.scatter(x_data, y_data, c='black', alpha=0.8, s=50)
            ax.text(0.1, 0.9, f'Pearson r$^2$: {str(pearsonR)}')
            ax.text(0.1, 0.85, f'Spearman $\\rho$: {str(spearmanR)}')
            ax.set_xlim([0.05, 0.95])
            ax.set_ylim([0.05, 0.95])
            ax.set_xlabel(screen_1_name)
            ax.set_ylabel(screen_2_name)

            out_fig_name = f'../../figs/screen_comparisons/screen_{screen_j}_vs_screen_{screen_i}'
            plt.savefig(out_fig_name +'.png')
            plt.savefig(out_fig_name +'.pdf')
            plt.savefig(out_fig_name +'.svg')

if __name__ == "__main__":
  main()
