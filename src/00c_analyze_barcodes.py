#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on December 30 17:10:38 2022
@author: alcantar
example usage: python 00c_analyze_barcodes.py -a
"""

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import collections
from collections import Counter
import matplotlib as mpl
from matplotlib import pyplot as plt

# enable pdfs to use helvetica font (among others)
from matplotlib import rc
plt.rcParams['pdf.fonttype'] = 42

def hamming_distance(sequence_1,
                    sequence_2):

    '''
    calculate hamming distance between two strings: that is, what is the minimal
    number of substitutions to get from string 1 to string 2. this script assumes
    that len(sequence_1) == len(sequence_2). this function is adapted from:
    http://claresloggett.github.io/python_workshops/improved_hammingdist.html

    PARAMETERS
    --------------------
    sequence_1: str
        first string or nucleic acid sequence to compare
    sequence_2: str
        second string or nucleic acid sequence to compare

    RETURNS
    --------------------
    distance: int
        the hamming distance between sequence_1 and sequence_2
    '''

    # initialize distance to 0
    distance = 0
    len_sequence_1 = len(sequence_1)

    # loop over entire string / sequence and compare each position of
    # sequence 1 vs. 2
    for base in range(len_sequence_1):
        # Add 1 to the distance if these two characters are not equal
        if sequence_1[base] != sequence_2[base]:
            distance += 1
    # return hamming distance
    return(distance)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='annotate hamming values?')
    args = parser.parse_args()

    # flag indicating whether to annotate heatmap with hamming distance values
    annotate_heatmap = args.a

    # define chromatin regulator to barcode sequence mappings
    CR_to_barcode_dict = {'ada2': 'CGATCCACA',
                              'arp6': 'TGTTGCAAG',
                              'arp9': 'TTGACCGAG',
                              'bre2': 'AACCGGAAG',
                              'btt1': 'GACAAGCAA',
                              'cac2': 'CAATCGGAG',
                              'cdc36': 'ACCATCGAA',
                              'DAM' : 'AGTAAGCCG',
                              'dpd4': 'GTCGACACA',
                              'eaf3': 'GCTCTAAGA',
                              'eaf5': 'CAGCATACA',
                              'empty': 'CTAGATCCG',
                              'esa1': 'TACAACAGG',
                              'gapdh': 'TCATTCGCG',
                              'gcn5': 'GTGGTTGTA',
                              'hat1': 'ATGCTTAGG',
                              'hat2': 'AGACTAGCA',
                              'hda1': 'AAGGAACAG',
                              'hhf2': 'GGTACCTAA',
                              'hmt2': 'TAGCTCGGA',
                              'hos1': 'GATGAGTGG',
                              'hos2': 'CCAGGTATA',
                              'hst2': 'ACATGACGG',
                              'htl1': 'GGCTATTGA',
                              'ies5': 'GTGTGCCAA',
                              'ldb7': 'GTAACTCGA',
                              'lge1': 'TATCCACGG',
                              'med4': 'GACTTCTAG',
                              'med7': 'CTCCGTTAA',
                              'mig1': 'CCGTTACCA',
                              'nhp6a': 'TGAGAGAGA',
                              'nhp10': 'CAGAGAGAA',
                              'nut2': 'ATCTAAGCG',
                              'pyk1': 'ATAAGCTCG',
                              'pyk2': 'CATGTAGCG',
                              'rpd3': 'GCTTATCAG',
                              'rtt102': 'CAAGTCCAA',
                              'sfh1': 'TAGGCGTAA',
                              'sir2': 'ATGCAGGAA',
                              'snf11': 'CGTCAACAA',
                              'spp1': 'CTGGCAAGA',
                              'srb7': 'TCACGGCAA',
                              'sus1': 'TCAACGTGG',
                              'swc5': 'GCAGAAGAA',
                              'swd3': 'AGTGGTGAA',
                              'tdh3': 'CGCTTGATG',
                              'vp16': 'CATTGGAGA',
                              'vps71': 'GTTCGAGAG',
                                }

    # extract all CR names
    CR_list = list(CR_to_barcode_dict.keys())

    # cluster CRs by chromatin regulating complex of protein class
    acetyltransferase_complex = ['hat1', 'hat2']
    deacetyltransferase_complex = ['hda1', 'hos1', 'hos2', 'hst2', 'sir2']
    rsc = ['htl1', 'ldb7', 'sfh1']
    swr1 = ['arp6', 'swc5', 'vps71']
    compass = ['bre2', 'spp1', 'swd3']
    mediator = ['med4', 'med7', 'nut2', 'srb7']
    saga = ['ada2', 'gcn5', 'sus1']
    kinase = ['gapdh', 'pyk1', 'pyk2', 'tdh3']
    methyltransferase = ['DAM', 'hmt2']
    swi_snf = ['arp9', 'rtt102', 'snf11']
    other = ['btt1', 'cac2', 'cdc36', 'dpd4', 'eaf3', 'eaf5', 'esa1', 'hhf2',
             'ies5','lge1', 'mig1', 'nhp6a','nhp10',  'rpd3', 'vp16', 'empty']

    CR_list_all = acetyltransferase_complex + compass + kinase + \
                    deacetyltransferase_complex + mediator + methyltransferase + \
                    rsc + saga +swi_snf + swr1 + other

    regulators_first_position = [CR for CR in CR_list_all if CR in CR_list]
    regulators_second_position = [CR for CR in CR_list_all if CR in CR_list]

    # initialize empty dataframe that will contain hamming distances between
    # all barcodes
    barcode_hamming_dist_df = pd.DataFrame(index=regulators_first_position, columns=regulators_second_position)


    # find hamming distance between all barcode pairs
    for first_CR in CR_list:
        first_CR_barcode = CR_to_barcode_dict[first_CR]
        for second_CR in CR_list:
            second_CR_barcode = CR_to_barcode_dict[second_CR]
            barcode_hamming_dist_df.at[first_CR,second_CR] = \
            int(hamming_distance(first_CR_barcode, second_CR_barcode))
    # need to fillna values, otherwise seaborn throws an error in the next step
    barcode_hamming_dist_df = barcode_hamming_dist_df.fillna(np.nan)

    # define plot parameters
    plt.figure(figsize=(20,20))
    plt.rcParams.update({'font.size': 18,
                         'font.family':'helvetica'})
    # upper and lower triangle are identical, so extract upper triangle to use
    # as a mask
    matrix = np.triu(barcode_hamming_dist_df)

    # define custom, discrete colormap
    custom_cmap = mpl.colors.ListedColormap(['#000000',
                                             '#FFFFFF',
                                             '#FFFFFF',
                                             '#FFFFFF',
                                             '#FFFFFF',
                                             '#C3CADE',
                                             '#A0B8D5',
                                             '#7AA6CA',
                                             '#3376AF',
                                             '#1B4A6F'])

    # create heatmap with all hamming distances
    heatmap = sns.heatmap(barcode_hamming_dist_df, cmap=custom_cmap,
                          xticklabels=True, yticklabels=True, square=True,
                          annot=annotate_heatmap, annot_kws={'size': 12},
                          mask=matrix)
    barcode_hamming_dist_df.to_csv('barcode_values.csv')

    plt.ylabel('Barcode ID', fontsize=18)
    plt.xlabel('Barcode ID', fontsize=18)

    plt.tight_layout()
    plt.savefig('../figs/barcode_hamming_distances.pdf')
    plt.savefig('../figs/barcode_hamming_distances.png')
    plt.savefig('../figs/barcode_hamming_distances.svg')

    # print summary of hamming distances -- to plot in PRISM
    barcode_hamming_dist_values = np.tril(barcode_hamming_dist_df.values)
    barcode_hamming_dist_values_flat = list(barcode_hamming_dist_values.flatten())
    barcode_hamming_dist_list = [x for x in barcode_hamming_dist_values_flat if x != 0]
    barcode_hamming_dist_list.sort()
    barcode_counts = Counter(barcode_hamming_dist_list)
    barcode_counts = dict(sorted(barcode_counts.items(), key=lambda item: item[0], reverse=False))

    print('Barcode hamming distances summary')
    print('----------------------------------')
    print(barcode_counts)

if __name__ == "__main__":
    main()
