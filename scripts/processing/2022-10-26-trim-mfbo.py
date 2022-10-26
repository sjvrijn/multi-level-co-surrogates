#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2022-10-26-trim-mfbo.py: utility processing script to trim back the data of
a run to a smaller budget than originally given. E.g. trimming a fixed-strategy
run with init_budget=100 back to as if it had been given init_budget=50.
"""

import argparse
import shutil

import numpy as np
import pandas as pd
from parse import compile
from pyprojroot import here


data_path = here('files/2020-11-05-simple-mfbo/')

EG_TEMPLATE = compile('errorgrid_{iteration:d}.nc')



def make_trimmed_copy(current_path, from_budget, new_path, to_budget):

    if to_budget >= from_budget:
        raise ValueError("Cannot trim to a larger budget")

    new_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(current_path / 'log.csv', sep=';')

    # Adjust 'budget left' values to new starting budget
    budget_left = df['budget']
    used_budget = from_budget - budget_left
    new_budget_left = to_budget - used_budget
    df['budget'] = new_budget_left

    df = df[df['budget'] >= 0]
    last_iter = np.max(df['iteration'].values)
    df.to_csv(new_path / 'log.csv', sep=';', index=False)

    for file in current_path.iterdir():
        if file.name == 'log.csv':
            continue
        if 'errorgrid' in file.name:
            match = EG_TEMPLATE.parse(file.name)
            if match['iteration'] > last_iter:
                continue
        shutil.copy(file, new_path / file.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exts",
                        help="")
    args = parser.parse_args()

