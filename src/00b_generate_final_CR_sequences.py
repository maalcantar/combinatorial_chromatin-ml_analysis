#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on October 24 11:31:35 2022
@author: alcantar
example usage: python 00b_generate_final_CR_sequences.py
"""

import pandas as pd
from Bio.Restriction import *
from Bio.Seq import Seq
# from Bio.Alphabet.IUPAC import IUPACAmbiguousDNA

def check_rd_sites(seq,
                   rd_prefix = 'aagacctattc',
                   rd_suffix = 'caatt',
                   amb = Seq(''),
                   combigem_rb = RestrictionBatch([BamHI,BbsI,BglII,EcoRI,MfeI, BsaI]),
                   stats=False):
    '''
    Check for the presence of restriction sites that would make the sequence
    incompatible with the CombiCR protocol

    PARAMETERS
    --------------------
    seq: str
        sequence to check for incompatible restriction sites
    rd_prefix: str
        region upsteam of seq that should be taken into account
        when checking for restriction sites
        [default is what is normally flanking barcode]
    rd_suffix: str
        region downstream of seq that should be taken into account
        when checking for restriction sites
        [default is what is normally flanking barcode]
    amb: DNA type
        dna type to convert sequence into
    combigem_rb: restriction enzyme list
        restriction sites to scan for
        [default is combiCR incompatible restriction enzymes]
    stats: boolean
        indicate whether to return stats on how many of each incompatible
        restriction sites are present.

    RETURNS
    --------------------
    total_num_rd_sites: int
        total number of incompatible restriction sites
    restriction_stats: dict list
        list of dictionaries indicating how many of each restriction site
        are present in the sequence

    '''
    # define full sequence to check, including flanking regions that
    # should be taken into account
    seq_full = rd_prefix + seq + rd_suffix
    seq_full_seq = Seq(seq_full)

    # search for combiCR-incompatible restriction sites
    rd_sites_list = []
    rd_sites_list.append(len(combigem_rb.search(seq_full_seq)[BamHI]))
    rd_sites_list.append(len(combigem_rb.search(seq_full_seq)[BbsI]))
    rd_sites_list.append(len(combigem_rb.search(seq_full_seq)[BglII]))
    rd_sites_list.append(len(combigem_rb.search(seq_full_seq)[EcoRI]))
    rd_sites_list.append(len(combigem_rb.search(seq_full_seq)[MfeI]))
    rd_sites_list.append(len(combigem_rb.search(seq_full_seq)[BsaI]))

    total_num_rd_sites = sum(rd_sites_list)

    # return number of each restriction site found, if requested
    # otherwise, just return total number of sites
    if stats:
        num_rd_sites = {'BamHI': rd_sites_list[0],
                       'BbsI': rd_sites_list[1],
                       'BglII': rd_sites_list[2],
                       'EcoRI': rd_sites_list[3],
                       'MfeI': rd_sites_list[4],
                       'BsaI': rd_sites_list[5]}
        return(total_num_rd_sites, num_rd_sites)
    else:
        return(total_num_rd_sites)


def main():
    # load barcode set and annotated CR dataframe
    barcode_seqs = pd.read_csv('../data/library_sequences/barcodes_len-9_dist-5.csv').rename(columns={'x':'barcodes'})
    CR_df = pd.read_csv('../data/library_sequences/CRs_final_df_annotated.csv').drop('Unnamed: 0', axis=1)

    # add additional features to CR dataframe
    CR_df['barcode'] = 'NNNNNNNNN'
    CR_df['barcoded_CDS'] = 'N'

    # define prefixes and suffixes (this will facilitate ordering gblocks)
    dna_prefix = 'tataGGTCTCtGGTTCT'.lower()
    dna_suffix_1 = 'GGTTCTTGAGTTTGGGTCTTCGAGAAGACCTATTC'.lower()
    dna_suffix_2 = 'cAATTgGAgacctata'.lower()

    # define barcodes for pilot dataset, and save results to a new dataframe
    # NOTE: this does not check if barcode has restriction site, so taf14 will
    # always return a barcode that has a BbsI restriction site
    barcode_counter = 0
    for idx in CR_df.index:
        if CR_df.loc[idx,'pilot'] == 1.0:
            CR_df.loc[idx,'barcode'] = barcode_seqs.loc[barcode_counter][0]
            CR_df.loc[idx,'barcoded_CDS'] = dna_prefix + CR_df.loc[idx,'dna_sequence'].lower() \
            + dna_suffix_1 + barcode_seqs.loc[barcode_counter][0].upper() + dna_suffix_2
            barcode_counter += 1


    CR_df.to_csv('../data/library_sequences/CR_pilot_barcoded_library.csv')

# define barcodes
    for idx in CR_df.index:
        if (CR_df.loc[idx,'include_in_library'] == 1.0 or CR_df.loc[idx,'include_in_library_part2'] == 1.0) \
        and (CR_df.loc[idx,'pilot'] != 1.0):
            barcode_tmp = barcode_seqs.loc[barcode_counter][0]

            total_rd_sites = check_rd_sites(barcode_tmp,
                       rd_prefix = 'aagacctattc',
                       rd_suffix = 'caatt',
                       amb = Seq(''),
                       combigem_rb = RestrictionBatch([BamHI,BbsI,BglII,EcoRI,MfeI, BsaI]),
                       stats=False)

            while total_rd_sites > 0:
                print(CR_df.loc[idx,'gene_name'])
                print('barcode ' + barcode_tmp + ' contains a restriction site')
                barcode_counter += 1
                barcode_tmp = barcode_seqs.loc[barcode_counter][0]
                total_rd_sites = check_rd_sites(barcode_tmp,
                       rd_prefix = 'aagacctattc',
                       rd_suffix = 'caatt',
                       amb = Seq(''),
                       combigem_rb = RestrictionBatch([BamHI,BbsI,BglII,EcoRI,MfeI, BsaI]),
                       stats=False)

            CR_df.loc[idx,'barcode'] = barcode_tmp
            CR_df.loc[idx,'barcoded_CDS'] = dna_prefix + CR_df.loc[idx,'dna_sequence'].lower() + dna_suffix_1 + barcode_seqs.loc[barcode_counter][0].upper() + dna_suffix_2
            barcode_counter += 1

            final_barcode_list = list(CR_df['barcode'])
            final_barcode_list = list(filter(lambda a: a != 'NNNNNNNNN', final_barcode_list))

    num_barcodes = len(final_barcode_list)
    num_unique_barcodes = len(set(final_barcode_list))

    if num_barcodes == num_unique_barcodes:
        print('all ' + str(num_barcodes) + ' barcodes are unique')
    else:
        print('there are non-unique barcodes')

    CR_libary_part2_df = CR_df.copy()[CR_df['barcode'] != 'NNNNNNNNN'].reset_index(drop=True)
    CR_libary_part2_df.to_csv('../data/library_sequences/CR_library_part2_barcoded.csv')

if __name__ =='__main__':
    main()
