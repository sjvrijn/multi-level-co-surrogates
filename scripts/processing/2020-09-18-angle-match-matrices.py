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


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

plot_dir = here('plots/2020-09-18-angle-match-matrices/', warn=False)
plot_dir.mkdir(exist_ok=True, parents=True)

kriging_path = here('files/2019-09-mse-nc/')
non_kriging_path = here('files/2019-11-12-different-surrogates/')


class ConfidenceInterval(namedtuple('ConfidenceInterval', 'mean se lower upper')):

    def __contains__(self, value):
        return self.lower < value < self.upper

    def __str__(self):
        lower = self.lower if self.lower is not None else self.mean - 1.96*self.se
        upper = self.upper if self.upper is not None else self.mean + 1.96*self.se
        return f'95% CI: {self.mean:.4f} +/- {1.96*self.se:.4f} {np.array([lower, upper])}: H0{" not" if 0 in self else ""} rejected'


class Comparison(IntEnum):
    NO_MATCH = 0
    CI_MATCH = 1
    SINGLE_MATCH = 2
    DOUBLE_MATCH = 3


def determine_match(CI1, CI2):
    # is the midpoint of one CI within the bounds of the other CI?
    covered_1 = CI1.mean in CI2
    covered_2 = CI2.mean in CI1

    if covered_1 and covered_2:
        return Comparison.DOUBLE_MATCH
    if covered_1 or covered_2:
        return Comparison.SINGLE_MATCH
    if CI1.lower in CI2 or CI1.upper in CI2:  # reverse is implied
        return Comparison.CI_MATCH
    return Comparison.NO_MATCH




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
            angle_compare[f_idx, surrogates.index(surr_name)] = determine_match(kriging_CI, non_kriging_CI)

    plt.imshow(angle_compare)
    plt.title("comparison with kriging")
    plt.xlabel('models')
    plt.xticks(range(num_models), labels=surrogates, rotation='vertical')
    plt.ylabel('functions')
    plt.yticks(range(num_functions), labels=all_models.groupby(['ndim', 'fname']).groups.keys())
    plt.tight_layout()
    plt.savefig(plot_dir / 'kriging_match_angles.pdf')


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
            angle_compare[idx1, idx2] = determine_match(CI1, CI2)
            angle_compare[idx2, idx1] = angle_compare[idx1, idx2]  # mirror for clarity

        plt.imshow(angle_compare)
        plt.title(f'{ndim}D {func_name}')
        plt.xticks(range(num_models), labels=surrogates, rotation='vertical')
        plt.yticks(range(num_models), labels=surrogates)
        plt.tight_layout()
        plt.savefig(plot_dir / f'model-match-angle-{ndim}D-{func_name}.pdf')


def get_CI(row):
    return ConfidenceInterval(row['deg'], None, row['deg_low'], row['deg_high'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regen-csv', action='store_true')
    args = parser.parse_args()

    kriging_angles = proc.get_gradient_angles(kriging_path, force_regen=args.regen_csv)
    non_kriging_angles = proc.get_gradient_angles(non_kriging_path, force_regen=args.regen_csv)

    plot_kriging_match_angles(kriging_angles, non_kriging_angles)
    plot_model_match_angles(kriging_angles, non_kriging_angles)
