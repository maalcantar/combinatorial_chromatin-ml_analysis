#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on March 3 11:06:02 2023
@author: alcantar
example run: python 08b_generalizability_test_stats.py
"""
# activate virtual enviroment before running script
# source activate combicr_ml

import scipy.stats
import pandas as pd
import statsmodels
from statsmodels.stats.multitest import multipletests
import statsmodels.stats.multitest as multi

def main():
  pvals_list = []
  generalizability_test_results_Rsquared_path = '../data/preconcat_linker_generalizability_test_results_Rsquared.csv'
  generalizability_test_results_pearson_path = '../data/preconcat_linker_generalizability_test_results_pearson.csv'
  generalizability_test_results_spearman_path = '../data/preconcat_linker_generalizability_test_results_spearman.csv'

  generalizability_test_results_Rsquared_df = pd.read_csv(generalizability_test_results_Rsquared_path)
  generalizability_test_results_pearson_df = pd.read_csv(generalizability_test_results_pearson_path)
  generalizability_test_results_spearman_df = pd.read_csv(generalizability_test_results_spearman_path)

  true_column_name = 'train_test_true'
  scramble_column_name = 'train_test_scrambled'
  mixed_column_name = 'train_true_test_scrambled'
  mixed_column_name_2 = 'train_scrambled_test_true'

  Rsquared_true = list(generalizability_test_results_Rsquared_df[true_column_name])
  Rsquared_scramble = list(generalizability_test_results_Rsquared_df[scramble_column_name])
  Rsquared_mixed = list(generalizability_test_results_Rsquared_df[mixed_column_name])
  Rsquared_mixed2 = list(generalizability_test_results_Rsquared_df[mixed_column_name_2])

  Pearson_true = list(generalizability_test_results_pearson_df[true_column_name])
  Pearson_scramble = list(generalizability_test_results_pearson_df[scramble_column_name])
  Pearson_mixed = list(generalizability_test_results_pearson_df[mixed_column_name])
  Pearson_mixed2 = list(generalizability_test_results_pearson_df[mixed_column_name_2])

  Spearman_true = list(generalizability_test_results_spearman_df[true_column_name])
  Spearman_scramble = list(generalizability_test_results_spearman_df[scramble_column_name])
  Spearman_mixed = list(generalizability_test_results_spearman_df[mixed_column_name])
  Spearman_mixed2 = list(generalizability_test_results_spearman_df[mixed_column_name_2])

  Rsquared_true_vs_scramble_pval = scipy.stats.mannwhitneyu(Rsquared_true, Rsquared_scramble)[1]
  Rsquared_true_vs_mixed_pval = scipy.stats.mannwhitneyu(Rsquared_true, Rsquared_mixed)[1]
  Rsquared_true_vs_mixed2_pval = scipy.stats.mannwhitneyu(Rsquared_true, Rsquared_mixed2)[1]
  Rsquared_scramble_vs_mixed_pval = scipy.stats.mannwhitneyu(Rsquared_scramble, Rsquared_mixed)[1]
  Rsquared_scramble_vs_mixed2_pval = scipy.stats.mannwhitneyu(Rsquared_scramble, Rsquared_mixed2)[1]
  Rsquared_mixed_vs_mixed2_pval = scipy.stats.mannwhitneyu(Rsquared_mixed, Rsquared_mixed2)[1]

  pvals_list.append(Rsquared_true_vs_scramble_pval)
  pvals_list.append(Rsquared_true_vs_mixed_pval)
  pvals_list.append(Rsquared_true_vs_mixed2_pval)
  pvals_list.append(Rsquared_scramble_vs_mixed_pval)
  pvals_list.append(Rsquared_scramble_vs_mixed2_pval)
  pvals_list.append(Rsquared_mixed_vs_mixed2_pval)

  # print(f'True vs scambled: {Rsquared_true_vs_scramble_pval}')
  # print(f'True vs mixed: {Rsquared_true_vs_mixed_pval}')
  # print(f'scrambled vs mixed: {Rsquared_scramble_vs_mixed_pval}')

  # Pearson_true_vs_scramble_pval = scipy.stats.mannwhitneyu(Pearson_true, Pearson_scramble)[1]
  # Pearson_true_vs_mixed_pval = scipy.stats.mannwhitneyu(Pearson_true, Pearson_mixed)[1]
  # Pearson_scramble_vs_mixed_pval = scipy.stats.mannwhitneyu(Pearson_scramble, Pearson_mixed)[1]
  Pearson_true_vs_scramble_pval = scipy.stats.mannwhitneyu(Pearson_true, Pearson_scramble)[1]
  Pearson_true_vs_mixed_pval = scipy.stats.mannwhitneyu(Pearson_true, Pearson_mixed)[1]
  Pearson_true_vs_mixed2_pval = scipy.stats.mannwhitneyu(Pearson_true, Pearson_mixed2)[1]
  Pearson_scramble_vs_mixed_pval = scipy.stats.mannwhitneyu(Pearson_scramble, Pearson_mixed)[1]
  Pearson_scramble_vs_mixed2_pval = scipy.stats.mannwhitneyu(Pearson_scramble, Pearson_mixed2)[1]
  Pearson_mixed_vs_mixed2_pval = scipy.stats.mannwhitneyu(Pearson_mixed, Pearson_mixed2)[1]

  pvals_list.append(Pearson_true_vs_scramble_pval)
  pvals_list.append(Pearson_true_vs_mixed_pval)
  pvals_list.append(Pearson_true_vs_mixed2_pval)
  pvals_list.append(Pearson_scramble_vs_mixed_pval)
  pvals_list.append(Pearson_scramble_vs_mixed2_pval)
  pvals_list.append(Pearson_mixed_vs_mixed2_pval)

  # print(f'True vs scambled: {Pearson_true_vs_scramble_pval}')
  # print(f'True vs mixed: {Pearson_true_vs_mixed_pval}')
  # print(f'scrambled vs mixed: {Pearson_scramble_vs_mixed_pval}')

  # Spearman_true_vs_scramble_pval = scipy.stats.mannwhitneyu(Spearman_true, Spearman_scramble)[1]
  # Spearman_true_vs_mixed_pval = scipy.stats.mannwhitneyu(Spearman_true, Spearman_mixed)[1]
  # Spearman_scramble_vs_mixed_pval = scipy.stats.mannwhitneyu(Spearman_scramble, Spearman_mixed)[1]
  Spearman_true_vs_scramble_pval = scipy.stats.mannwhitneyu(Spearman_true, Spearman_scramble)[1]
  Spearman_true_vs_mixed_pval = scipy.stats.mannwhitneyu(Spearman_true, Spearman_mixed)[1]
  Spearman_true_vs_mixed2_pval = scipy.stats.mannwhitneyu(Spearman_true, Spearman_mixed2)[1]
  Spearman_scramble_vs_mixed_pval = scipy.stats.mannwhitneyu(Spearman_scramble, Spearman_mixed)[1]
  Spearman_scramble_vs_mixed2_pval = scipy.stats.mannwhitneyu(Spearman_scramble, Spearman_mixed2)[1]
  Spearman_mixed_vs_mixed2_pval = scipy.stats.mannwhitneyu(Spearman_mixed, Spearman_mixed2)[1]

  pvals_list.append(Spearman_true_vs_scramble_pval)
  pvals_list.append(Spearman_true_vs_mixed_pval)
  pvals_list.append(Spearman_true_vs_mixed2_pval)
  pvals_list.append(Spearman_scramble_vs_mixed_pval)
  pvals_list.append(Spearman_scramble_vs_mixed2_pval)
  pvals_list.append(Spearman_mixed_vs_mixed2_pval)

  # print(f'True vs scambled: {Spearman_true_vs_scramble_pval}')
  # print(f'True vs mixed: {Spearman_true_vs_mixed_pval}')
  # print(f'scrambled vs mixed: {Spearman_scramble_vs_mixed_pval}')

  padj_df = multi.multipletests(pvals_list, alpha=0.05, method='fdr_bh')
  print(padj_df)

if __name__ == "__main__":
  main()
