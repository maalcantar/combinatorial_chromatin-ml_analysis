#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on October 29 15:51:06 2022
@author: alcantar
example usage: python create_barcode_mutations_dict.py
"""

import collections
import pickle

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

def main():
    # initialize dictionary with CR: barcode mappings
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

    # create barcode to CR mappings
    barcode_to_CR = invert_dictionary(CR_to_barcode_dict)
    barcode_to_CR_mutations = barcode_to_CR.copy()

    nt_list = ['A', 'T', 'C', 'G', 'N']

    # loop through all barcodes
    for barcode, CR in barcode_to_CR.items():
        # loop through each position in barcode
        for nt_position in range(len(barcode)):
            # from nt_list remove the nt that is at the current position in the true barcode
            nt_tmp = barcode[nt_position]
            nt_list_tmp =  nt_list.copy()
            nt_list_tmp.remove(nt_tmp)
            # make all possible 1bp mutations
            for possible_snp in nt_list_tmp:
                barcode_mutated = list(barcode)
                barcode_mutated[nt_position] = possible_snp
                barcode_mutated = ''.join(barcode_mutated)
                # update new dict with all possible mutations
                barcode_to_CR_mutations.update({barcode_mutated: CR})

    # save new dictionary as .pkl file
    pickle_output_path ='../../data/combicr_barcode_mutations.pkl'

    with open(pickle_output_path, 'wb') as handle:
        pickle.dump(barcode_to_CR_mutations, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
