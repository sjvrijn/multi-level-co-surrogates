#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-02-19-adjustable-gradients.py: script to create scatterplots of the
gradients in the error versus the actual correlation between high- and low-
fidelity functions. Creates a csv of the all relevant values as intermediate
step to speed up any potential rerun.
"""

import argparse
from collections import namedtuple
from enum import Enum
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


class Comparison(Enum):
    NO_MATCH = 0
    CI_MATCH = 1
    SINGLE_MATCH = 2
    DOUBLE_MATCH = 3


def determine_match(CI1, CI2):
    # is the midpoint of one CI within the bounds of the other CI?
    covered_1 = CI1.mid in CI2
    covered_2 = CI2.mid in CI1

    if covered_1 and covered_2:
        return Comparison.DOUBLE_MATCH
    if covered_1 or covered_2:
        return Comparison.SINGLE_MATCH
    if any(bound in CI2 for bound in CI1.bounds):  # reverse is implied
        return Comparison.CI_MATCH
    return Comparison.NO_MATCH




def plot_kriging_match_angles(df_kriging, df_non_kriging):
    pass


    # angle_compare = np.zeros((len(F), len(M - 1)))
    # for func_name in F:
    #     CI_kriging = L[func_name, kriging].get_angle_CI()
    #     for model in M:
    #         CI = group[m1].get_angle_CI()
    #         angle_compare[f, model] = determine_match(CI_kriging, CI)
    #
    #     plt.imshow(angle_compare, title="comparison with kriging")


def plot_model_match_angles(df):
    pass
    # for func_name, group in L.groupby(F):
    #     angle_compare = np.zeros((len(M), len(M)))
    #     for m1, m2 in combinations(M):
    #         CI1 = group[m1].get_angle_CI()
    #         CI2 = group[m2].get_angle_CI()
    #         angle_compare[m1, m2] = determine_match(CI1, CI2)
    #         angle_compare[m2, m1] = angle_compare[m1, m2]  # mirror for clarity
    #
    #     plt.imshow(angle_compare, title=func_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--regen-csv', action='store_true')

    args = parser.parse_args()

    kriging_angles = proc.get_gradient_angles(kriging_path, force_regen=args.regen_csv)
    non_kriging_angles = proc.get_gradient_angles(non_kriging_path, force_regen=args.regen_csv)

    # plot_kriging_match_angles(kriging_angles, non_kriging_angles)
