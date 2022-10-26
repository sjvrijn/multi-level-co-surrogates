#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2022-10-26-trim-mfbo.py: utility processing script to trim back the data of
a run to a smaller budget than originally given. E.g. trimming a fixed-strategy
run with init_budget=100 back to as if it had been given init_budget=50.
"""

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from parse import compile
from pyprojroot import here


data_path = here('files/2020-11-05-simple-mfbo/')

EG_TEMPLATE = compile('errorgrid_{iteration:d}.nc')
FOLDER_FORMAT = '{part1}-b{init_budget:d}-{part2}'
FOLDER_TEMPLATE = compile(FOLDER_FORMAT)



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

    # Trim back data and save new file
    df = df[df['budget'] >= 0]
    last_iter = np.max(df['iteration'].values)
    df.to_csv(new_path / 'log.csv', sep=';', index=False)

    # copy all remaining files, except errorgrids that should be trimmed
    for file in current_path.iterdir():
        if file.name == 'log.csv':
            continue
        if 'errorgrid' in file.name:
            match = EG_TEMPLATE.parse(file.name)
            if match['iteration'] > last_iter:
                continue
        shutil.copy(file, new_path / file.name)


def main(args):
    for folder in args.base_folder.glob(args.selector):
        match = FOLDER_TEMPLATE.parse(folder.name)
        if not match:
            continue
        old_budget = match['init_budget']
        new_folder = args.base_folder / FOLDER_FORMAT.format(part1=match['part1'], init_budget=args.budget, part2=match['part2'])

        make_trimmed_copy(folder, old_budget, new_folder, args.budget)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("selector", type=str,
                        help="glob search string, used to select folders")
    parser.add_argument("budget", type=int,
                        help="what the 'original' init_budget should be after trimming")
    parser.add_argument("--base-folder", type=Path, default=data_path,
                        help="Path in which to search for data to trim. Default: files/2020-11-05-simple-mfbo/")
    args = parser.parse_args()

    main(args)
