#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on October 18 15:03:13 2022
@author: alcantar
example usage 1: python 02_count_barcode.py -i ../../combicr_data/combicr_outputs/maa-004/barcodes/ -m ../../combicr_data/Quintata_NGS_raw/MAA_004/HiSeq_2022-10-26/Fastq/MAA_004_metadata.txt
"""
import argparse

import os
import sys

import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import glob
import difflib
import collections
from collections import Counter
from Bio.Restriction import *
from Bio.Seq import Seq
from Bio import SeqIO

import pickle
import random

def invert_dictionary(dict_to_invert):

    '''
    invert dictionary: that is, keys become values and vice-versa.
    adapted from:
    https://stackoverflow.com/questions/8377998/swapping-items-in-a-dictionary-and-order

    PARAMETERS
    --------------------
    dict_to_invert: dict
        dictionary to be inverted

    RETURNS
    --------------------
    inverted_dictionary: dict
        dictionary that has been inverted
    '''

    inverted_dictionary = collections.OrderedDict((dict_to_invert[key_val], \
    key_val) for key_val in sorted(dict_to_invert))

    return(inverted_dictionary)

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

def count_barcodes(barcodes_dir,
                    metadata_path,
                    premade_dict_flag):

    '''
    count barcodes for each samples and match barcodes to their corresponding
    chromatin regulator (CR) combination

    PARAMETERS
    --------------------
    barcodes_dir: str
        path to directory containing barcodes extracted for each read
    metadata_path: str
        path to sample metadata
    premade_dict_flag: boolean / int [0, 1]
        indicate whether to use preade bacode dictionary [1] or use the closest
        match method to count barcodes

    RETURNS
    --------------------
    barcode_count_tracker: dict
        dictionary containing CR occurances in each sample
    '''

    # sort barcode files and read in metadata
    barcode_files_list = sorted(glob.glob(barcodes_dir + '*.txt'))
    # handle naming convention that starts with a letter (e.g., A01 vs 01)
    try:
        barcode_files_list.sort(key=lambda x: int(x.split('/')[-1].split('_')[0]))
    except ValueError:
        letter_to_val_dict = {'A': '1',
                              'B':'2',
                              'C':'3',
                              'D':'4',
                              'E':'5',
                              'D':'6',
                              'F':'7',
                              'G':'8'}

        # this line just replaces the letter in the prefix with a number (e.g.,
        # A becomes 1, B becomes 2). this helps with sorting the files
        # barcode_files_list.sort(key=lambda x: int(x.split('/')[-1].split('_')[0][1:]))
        barcode_files_list.sort(key=lambda x: int(x.split('/')[-1].split('_')[0].replace(x.split('/')[-1].split('_')[0][0], letter_to_val_dict[x.split('/')[-1].split('_')[0][0]])))
    sample_metadata_df = pd.read_csv(metadata_path, sep='\t')

    if premade_dict_flag:
        path_to_premade_dict_pkl = '../data/combicr_barcode_mutations.pkl'
        with open(path_to_premade_dict_pkl, 'rb') as handle:
            barcode_to_CR = pickle.load(handle)

    else:

        # dictionary of CR-to-barcode conversions
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

        # dictionary of barcode-to-CR conversions
        barcode_to_CR = invert_dictionary(CR_to_barcode_dict)
        barcode_list = list(barcode_to_CR.keys())

    barcode_count_tracker = dict()

    # define colnames for barcode files
    colnames=['read_name', 'start_pos', 'end_post', 'DB', 'score', 'strand', 'barcode']

    # loop through all barcode files and extract corresponding CR combinations
    for sample in tqdm(barcode_files_list):

        sample_name_prefix = sample.split('/')[-1].replace('_barcodes.txt','')
        sample_name_tmp = sample.split('/')[-1].replace('barcodes.txt','R1_001.fastq.gz')

        metadata_tmp = sample_metadata_df[sample_metadata_df.copy()['sample_name'] == sample_name_tmp]
        experiment_name = list(metadata_tmp['experiment_name'])[0]
        # sample_type = list(metadata_tmp['sample_type'])[0]

        # initialize lists for calling barcodes and storing CRs
        CR1_list =[]
        CR2_list = []
        CR_combo = []

        # read in barcodes from barcode file
        barcodes_list = [barcode_data.split('\t')[-1].rstrip('\n') for barcode_data in open(sample,'r').readlines()]

        for barcode_tmp in tqdm(barcodes_list):
            barcode1_tmp = barcode_tmp[-9:] # barcode for CR that binds closest to promoter
            barcode2_tmp = barcode_tmp[0:9] # barcode for CR that binds furthest from promoter

            # match barcode to CR. if not a perfect match, find the closest match. if
            # the closest match has a hamming distance <=1, then accept the new match.
            # otherwise, discard the read.

            if premade_dict_flag:
                try:
                    barcode1_CR_match = barcode_to_CR[barcode1_tmp]
                    barcode2_CR_match = barcode_to_CR[barcode2_tmp]
                except KeyError:
                    barcode1_CR_match = 'NO_MATCH'
                    barcode2_CR_match = 'NO_MATCH'
                    continue
            else:
                try:
                    barcode1_CR_match = barcode_to_CR[barcode1_tmp]
                except KeyError:
                    try:
                        barcode1_match = difflib.get_close_matches(barcode1_tmp, barcode_list,1)[0]
                        hamming_barcode_1 = hamming_distance(barcode1_tmp, barcode1_match)
                        if hamming_barcode_1 <=1:
                              barcode1_CR_match = barcode_to_CR[barcode1_match]
                        else:
                            continue
                    except IndexError:
                        barcode1_CR_match = 'NO_MATCH'
                        continue

                try:
                    barcode2_CR_match = barcode_to_CR[barcode2_tmp]
                except KeyError:
                    try:
                        barcode2_match = difflib.get_close_matches(barcode2_tmp, barcode_list,1)[0]
                        hamming_barcode_2 = hamming_distance(barcode2_tmp, barcode2_match)
                        if hamming_barcode_2 <=1:
                              barcode2_CR_match = barcode_to_CR[barcode2_match]
                        else:
                            continue
                    except IndexError:
                        barcode2_CR_match = 'NO_MATCH'
                        continue

            # update lists keeping track of barcode-CR matches
            CR1_list.append(barcode1_CR_match)
            CR2_list.append(barcode2_CR_match)
            CR_combo.append(barcode1_CR_match + '_'+ barcode2_CR_match)

        # count up the number of occurences for each CR
        CR1_list_counts = Counter(CR1_list)
        CR1_counts = dict(sorted(CR1_list_counts.items(), key=lambda item: item[1], reverse=True))
        CR2_list_counts = Counter(CR2_list)
        CR2_counts = dict(sorted(CR2_list_counts.items(), key=lambda item: item[1], reverse=True))
        CR_combo_counts = Counter(CR_combo)
        CR_combo_counts = dict(sorted(CR_combo_counts.items(), key=lambda item: item[1], reverse=True))

        barcode_count_tracker.update({sample_name_prefix:{'CR1_counts': CR1_counts,
                                                          'CR2_counts': CR2_counts,
                                                          'CR_combo_counts':CR_combo_counts}})

    pickle_dir = '../../combiCR_data/combiCR_outputs/' + experiment_name + '/pickle_files/'
    pickle_output_path = pickle_dir + 'combiCR_results.pkl'
    # create ouput folders if needed
    CHECK_FOLDER_MAIN = os.path.isdir(pickle_dir)
    if not CHECK_FOLDER_MAIN:
        os.makedirs(pickle_dir)
        print(f"created folder : {pickle_dir}")

    with open(pickle_output_path, 'wb') as handle:
        pickle.dump(barcode_count_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return(barcode_count_tracker)

def plot_CR_count_dist(CR_list,
                        sample_name,
                        count_type,
                        out_dir,
                        frequency=False,
                        save=True):

    '''
    creates a distribution plot showing the abundance of each CR or CR combination.
    this can be either frequency or raw read counts.

    PARAMETERS
    --------------------
    CR_list: list of strings
        list of CRs obtained by identifying barcodes. should be a list; not a dictionary
    sample_name: str
        sample name -- this is basically the prefix for the fastq.gz file
        this is used to correctly name the resulting plots
    count_type: str
        the type of CR list being plotted (e.g., CR1 list or CR combos)
        this is used to correctly name the resulting plots
    out_dir: str
        output directory
    frequency: boolean
        indicate whether to plot y-axis in terms of frequency (True) or raw
        read counts (False)
    save: boolean
        indicate whether to save plot

    RETURNS
    --------------------
    None -- just saves plots to specified output directory
    '''

    plt.figure()
    plt.rcParams.update({'font.size': 16,
                        'font.family':'sans-serif'})

    # CR_counts = Counter(CR_list)
    # CR_counts = dict(sorted(CR_counts.items(), key=lambda item: item[1], reverse=True))
    CR_counts=CR_list
    # convert counts to frequency if needed
    if frequency:
        CR_counts = {k: v / total for total in (sum(CR_counts.values()),) for k, v in CR_counts.items()}
        ylabel = 'Frequency'
    else:
        ylabel = 'Counts'
    df = pd.DataFrame.from_dict(CR_counts, orient='index')
    ax1 = df.plot(kind='bar', width=0.7,figsize=(16,8),color='cornflowerblue', edgecolor='black',legend=None)
    ax1.set_ylabel(ylabel,fontdict={'fontsize':16})
    ax1.set_xlabel(count_type, fontdict={'fontsize':16})

    # include number of combinations found
    if count_type == 'CR_combos':
        num_combinations = str(len(set(CR_list)))
        num_combinations_text = num_combinations + ' combinations'
        ax1.text(0.75, 0.75, num_combinations_text, horizontalalignment='center',
        verticalalignment='center', transform=ax1.transAxes)

    plt.tight_layout()

    plt.savefig(out_dir+sample_name+ '_' + count_type + '.pdf')
    plt.savefig(out_dir+sample_name+ '_' + count_type + '.png')
    plt.close('all')

def CR_abundances_from_barcodes(barcode_count_tracker,
                                metadata_path):

    '''
    create barplots showing how often a CR or CR combination appeared in sequencing run

    PARAMETERS
    --------------------
    barcode_count_tracker: dictionary of dictionaries
        dictionary containing list of all barcodes that appeared in each read
        this dictionry is broken down by sample; each sample contains list(s) of
        the identified barcodes
        NOTE: this is the output from identify_and_count_barcodes
    metadata_path: str
        path to metadata

    RETURNS
    --------------------
    none -- just plots stuff

    '''
    sample_names_in_dict = barcode_count_tracker.keys()
    sample_metadata = pd.read_csv(metadata_path, sep='\t')
    for sample in sample_names_in_dict:
        sample_file_name = sample + '_R1_001.fastq.gz'
        print(sample_file_name)
        metadata_tmp = sample_metadata[sample_metadata.copy()['sample_name'] == sample_file_name]
        sample_type = list(metadata_tmp['sample_type'])[0]
        experiment_name = list(metadata_tmp['experiment_name'])[0]

        figs_dir = '../../combiCR_data/combiCR_outputs/' + experiment_name + '/figs/'
        # create ouput folders if needed
        CHECK_FOLDER_MAIN = os.path.isdir(figs_dir)
        if not CHECK_FOLDER_MAIN:
            os.makedirs(figs_dir)
            print(f"created folder : {figs_dir}")


        CR_list1 = barcode_count_tracker[sample]['CR1_counts']
        CR_list2 = barcode_count_tracker[sample]['CR2_counts']
        CR_combo_list = barcode_count_tracker[sample]['CR_combo_counts']


        plot_CR_count_dist(CR_list=CR_list1,
                           sample_name=sample,
                           count_type='CR1_list',
                           out_dir=figs_dir,
                           frequency=False,
                           save=True)
        plot_CR_count_dist(CR_list=CR_list2,
                           sample_name=sample,
                           count_type='CR2_list',
                           out_dir=figs_dir,
                           frequency=False,
                           save=True)

        plot_CR_count_dist(CR_list=CR_combo_list,
                           sample_name=sample,
                           count_type='CR_combos',
                           out_dir=figs_dir,
                           frequency=False,
                            save=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='directory containing barcode files ')
    parser.add_argument('-m', help='path to metadata')
    # parser.add_argument('-b', help='target number of sequences when bootstrapping', default=0)
    parser.add_argument('--p', help='path to pickle')
    parser.add_argument('--d', help='use pre-made dictionary', default=0)
    args = parser.parse_args()

    barcodes_dir = args.i
    metadata_path = args.m
    premade_dict_flag = int(args.d)

    if args.p is None:
        # ensure that the barcode directory ends in a forward slash, for consistency
        if barcodes_dir[-1] != '/':
            barcodes_dir = barcodes_dir +'/'
        barcode_count_tracker = count_barcodes(barcodes_dir,
                                               metadata_path,
                                               premade_dict_flag)
    else:
        pickle_path=args.p
        print('Grabbing pickle file: ' + pickle_path)
        with open(pickle_path, 'rb') as handle:
            barcode_count_tracker = pickle.load(handle)

    CR_abundances_from_barcodes(barcode_count_tracker,
                                    metadata_path)
if __name__ == "__main__":
    main()
