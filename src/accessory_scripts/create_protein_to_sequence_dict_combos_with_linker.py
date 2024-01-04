#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on February 23 11:29:01 2023
@author: alcantar
example usage: python create_protein_to_sequence_dict_combos_with_linker.py
"""

import pickle
import random

def main():

    # get protein-sequence relationships from previous script output
    protein_name_to_aa_sequence_path = '../../data/combicr_protien_to_aa_sequence_dict.pkl'
    with open(protein_name_to_aa_sequence_path, 'rb') as handle:
        protein_name_to_aa_dict = pickle.load(handle)

    # add empty regulator to dictionary
    protein_name_to_aa_dict.update({'empty':''})

    # initialize dataframes that will contain concatenated protein sequences
    protein_combos_for_embeddings = dict()
    protein_combos_for_embeddings_scrambled = dict()

    linker_aa_seq = 'GGGGSGGGGS' # 10aa Gly-Ser linker 
    for protein_name_1 in protein_name_to_aa_dict:
        for protein_name_2 in protein_name_to_aa_dict:
            combo = protein_name_1 + '_' + protein_name_2
            combo_seq = protein_name_to_aa_dict[protein_name_1] + linker_aa_seq + protein_name_to_aa_dict[protein_name_2]
            combo_seq_scramble = list(combo_seq)

            # scramble sequence
            random.Random(42).shuffle(combo_seq_scramble)
            combo_seq_scramble = ''.join(combo_seq_scramble)

            protein_combos_for_embeddings.update({combo: combo_seq})
            protein_combos_for_embeddings_scrambled.update({combo: combo_seq_scramble})

    # define output paths
    pickle_output_path ='../../data/combicr_protein_to_aa_sequence_combos_linker_dict.pkl'
    pickle_output_scrambled_path ='../../data/combicr_protein_to_aa_sequence_combos_linker_scrambled_dict.pkl'

    # output non-scrambled sequences
    with open(pickle_output_path, 'wb') as handle:
        pickle.dump(protein_combos_for_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # output scrambled sequences
    with open(pickle_output_scrambled_path, 'wb') as handle:
        pickle.dump(protein_combos_for_embeddings_scrambled, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
