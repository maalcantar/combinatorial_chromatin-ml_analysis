#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 16 22:38:37 2023
@author: alcantar
example usage: python create_protein_to_sequence_dict.py
"""

import collections
import pandas as pd
import pickle

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
                          'dpd4': 'GTCGACACA', # typo (dpb4; keeping for consistency)
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
                          'hmt2': 'TAGCTCGGA', # typo (hmt1; keeping for consistency)
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

    # load in dataframe with all combicr protein information
    protein_annotation_df = pd.read_csv('../../data/library_sequences/CR_library_part2_barcoded.csv',index_col=0)
    protein_to_aa_sequence_dict = dict()

    # protein name-to-amino acid sequence relationships will be stored
    # in both dictionary (pickle) and fasta format
    with open('../../data/combicr_aa_sequence.fasta','w+') as aa_fasta:
        for protein_name in CR_to_barcode_dict:
            new_entry = '>' + protein_name
            aa_fasta.write(new_entry)
            aa_fasta.write('\n')
            if protein_name == 'empty':
                continue
            # account for typos in original dataframe
            if protein_name == 'dpd4':
                current_protein_row = protein_annotation_df.copy()[protein_annotation_df['gene_name']=='dpb4']
            elif protein_name == 'DAM':
                current_protein_row = protein_annotation_df.copy()[protein_annotation_df['gene_name']=='dam']
            elif protein_name == 'hmt2':
                current_protein_row = protein_annotation_df.copy()[protein_annotation_df['gene_name']=='hmt1']
            else:
                current_protein_row = protein_annotation_df.copy()[protein_annotation_df['gene_name']==protein_name]
            current_protein_aa_sequence = list(current_protein_row['aa_sequence'])[0]
            if current_protein_aa_sequence[-1] == '*':
                current_protein_aa_sequence=current_protein_aa_sequence[:-1]
                print(current_protein_aa_sequence[-1])
            protein_to_aa_sequence_dict.update({protein_name: current_protein_aa_sequence})
            aa_fasta.write(current_protein_aa_sequence)
            aa_fasta.write('\n')

    # save new dictionary as .pkl file
    pickle_output_path ='../../data/combicr_protien_to_aa_sequence_dict.pkl'
    #
    with open(pickle_output_path, 'wb') as handle:
        pickle.dump(protein_to_aa_sequence_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
