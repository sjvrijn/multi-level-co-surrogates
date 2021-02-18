#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-09-18-angle-match-matrices.py: compare the measured angles
from using different surrogate models, and plot them as 'heatmaps'.
"""

import argparse
from collections import namedtuple
from enum import IntEnum
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyprojroot import here

import processing as proc

print(f'Running script: {__file__}')


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

plot_dir = here('plots/2020-09-18-angle-match-matrices/', warn=False)
plot_dir.mkdir(exist_ok=True, parents=True)

kriging_path = here('files/2019-09-mse-nc/')
non_kriging_path = here('files/2019-11-12-different-surrogates/')



def plot_kriging_match_angles(df_kriging, df_non_kriging):
    all_models = df_kriging.append(df_non_kriging)\
        .drop(columns=['category', 'param'])

    surrogates = list(all_models['surrogate'].unique())
    num_models = len(surrogates)
    num_functions = all_models.groupby(['ndim', 'fname']).ngroups

    angle_compare = np.full((num_functions, num_models), np.nan)
    for f_idx, ((ndim, fname), sub_df) in enumerate(all_models.set_index('surrogate').groupby(['ndim', 'fname'])):
        kriging_CI = get_CI(sub_df.loc['Matern'])
        for surr_name, row in sub_df.iterrows():
            non_kriging_CI = get_CI(row)
            angle_compare[f_idx, surrogates.index(surr_name)] = proc.determine_match(kriging_CI, non_kriging_CI)

    plt.imshow(angle_compare)
    plt.title("comparison with kriging")
    plt.xlabel('models')
    plt.xticks(range(num_models), labels=surrogates, rotation='vertical')
    plt.ylabel('functions')
    plt.yticks(range(num_functions), labels=all_models.groupby(['ndim', 'fname']).groups.keys())
    plt.tight_layout()
    for ext in proc.extensions:
        plt.savefig(plot_dir / f'kriging_match_angles.{ext}', dpi=300)


def plot_model_match_angles(df_kriging, df_non_kriging):
    all_models = df_kriging.append(df_non_kriging)\
        .drop(columns=['category', 'param'])

    surrogates = list(all_models['surrogate'].unique())
    num_models = len(surrogates)


    for (ndim, func_name), group in all_models.set_index('surrogate').groupby(['ndim', 'fname']):
        angle_compare = np.full((num_models, num_models), np.nan)
        for m1, m2 in combinations(surrogates, 2):
            idx1, idx2 = surrogates.index(m1), surrogates.index(m2)
            try:
                CI1 = get_CI(group.loc[m1])
                CI2 = get_CI(group.loc[m2])
            except KeyError:
                continue
            angle_compare[idx1, idx2] = proc.determine_match(CI1, CI2)
            angle_compare[idx2, idx1] = angle_compare[idx1, idx2]  # mirror for clarity

        plt.imshow(angle_compare)
        plt.title(f'{ndim}D {func_name}')
        plt.xticks(range(num_models), labels=surrogates, rotation='vertical')
        plt.yticks(range(num_models), labels=surrogates)
        plt.tight_layout()
        for ext in proc.extensions:
            plt.savefig(plot_dir / f'model-match-angle-{ndim}D-{func_name}.{ext}', dpi=300)


def get_CI(row):
    return proc.ConfidenceInterval(row['deg'], None, row['deg_low'], row['deg_high'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-regen', action='store_true')
    args = parser.parse_args()

    kriging_angles = proc.get_gradient_angles(kriging_path, force_regen=args.force_regen)
    non_kriging_angles = proc.get_gradient_angles(non_kriging_path, force_regen=args.force_regen)

    plot_kriging_match_angles(kriging_angles, non_kriging_angles)
    plot_model_match_angles(kriging_angles, non_kriging_angles)
