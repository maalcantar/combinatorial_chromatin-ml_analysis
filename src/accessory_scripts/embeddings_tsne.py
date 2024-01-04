#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 20 12:48:50 2023
@author: alcantar
example run: python embeddings_tsne.py -i ../../data/unirep_embeddings_raw/unirep64/combicr_unirep_embeddings_average_hidden_64_dict.pkl
"""
# activate virtual enviroment before running script
# source activate combicr_ml

import pandas as pd
import pickle
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help='path to embeddings pickle')
    args = parser.parse_args()

    embeddings_path = args.i

    CR_to_class_df = {'ada2': 'SAGA',
                          'arp6': 'SWR1',
                          'arp9': 'SWI/SNF', #  and RSC
                          'bre2': 'COMPASS',
                          'btt1': 'Other',
                          'cac2': 'Other', # CAF-1
                          'cdc36': 'Other', # CCR4-NOT
                          'DAM' : 'Methyltransferase',
                          'dpd4': 'Other', # DNA pol epsilon and of ISW2
                          'eaf3': 'Other',
                          'eaf5': 'Other', # NuA4
                          'empty': 'Other',
                          'esa1': 'Other',
                          'gapdh': 'Glycolytic kinase',
                          'gcn5': 'SAGA',
                          'hat1': 'Acetyltransferase complex',
                          'hat2': 'Acetyltransferase complex',
                          'hda1': 'Deacetylase complex',
                          'hhf2': 'Other', # Histone H4
                          'hmt2': 'Methyltransferase', # hmt1
                          'hos1': 'Deacetylase complex',
                          'hos2': 'Deacetylase complex',
                          'hst2': 'Deacetylase complex',
                          'htl1': 'RSC',
                          'ies5': 'Other', # INO80
                          'ldb7': 'RSC',
                          'lge1': 'Other', #histone uniquitination
                          'med4': 'MEDiator',
                          'med7': 'MEDiator',
                          'mig1': 'Other',
                          'nhp6a': 'Other',
                          'nhp10': 'Other',
                          'nut2': 'MEDiator',
                          'pyk1': 'Glycolytic kinase',
                          'pyk2': 'Glycolytic kinase',
                          'rpd3': 'Other',
                          'rtt102': 'SWI/SNF', # and RSC
                          'sfh1': 'RSC',
                          'sir2': 'Deacetylase complex',
                          'snf11': 'SWI/SNF',
                          'spp1': 'COMPASS', # methylation
                          'srb7': 'MEDiator',
                          'sus1': 'SAGA',
                          'swc5': 'SWR1',
                          'swd3': 'COMPASS',
                          'tdh3': 'Glycolytic kinase',
                          'vp16': 'Other',
                          'vps71': 'SWR1',
                            }

    # colors will be modified in Illustrator
    class_to_color_dict = {"Other": '#000000',
                            "SAGA": '#F8E0DA',
                            "SWR1": '#9ED5B7',
                            "SWI/SNF": '#8E8E8E',
                            "COMPASS": '#FBDE92',
                            "Methyltransferase": '#A04D4E',
                            "Glycolytic kinase": '#DC3A78',
                            "Acetyltransferase complex": '#055275',
                            "Deacetylase complex": '#A4DFF9',
                            "RSC": '#089099',
                            "MEDiator": '#E88470'}

    with open(embeddings_path,'rb') as handle:
        embeddings_dict = pickle.load(handle)

    embeddings_final = pd.DataFrame(embeddings_dict).T

    X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=7, random_state=2).fit_transform(embeddings_final)
    plt.figure(figsize=(8,8))
    plt.rcParams["figure.figsize"] = [8, 8]
    plt.rcParams["figure.dpi"] = 400

    X_embedded_df = pd.DataFrame(X_embedded).rename(dict(enumerate(embeddings_final.index)))
    colors_list = []
    for reg in X_embedded_df.index:
        reg_class = CR_to_class_df[reg]
        reg_color = class_to_color_dict[reg_class]
        colors_list.append(reg_color)
    X_embedded_df['colors'] = colors_list
    ax = X_embedded_df.plot.scatter(x=0, y=1, alpha=1.0, c='colors', s=150,edgecolors='black')
    # Annotate each data point
    for i, txt in enumerate(X_embedded_df.index):
        ax.annotate(txt, (X_embedded_df[0].iat[i]+0.05, X_embedded_df[1].iat[i]))
    plt.savefig('../../figs/machine_learning/dimensionality/unirep64_embeddings_tsne.pdf')
    plt.savefig('../../figs/machine_learning/dimensionality/unirep64_embeddings_tsne.png')
    plt.savefig('../../figs/machine_learning/dimensionality/unirep64_embeddings_tsne.svg')

    plt.rcParams["figure.figsize"] = [8, 8]
    plt.rcParams["figure.dpi"] = 400

    X_embedded_df.to_csv('../../data/embeddings_tSNE_df.csv')


if __name__ == "__main__":
    main()
