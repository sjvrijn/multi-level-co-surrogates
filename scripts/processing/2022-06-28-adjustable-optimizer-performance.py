#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2022-06-28-adjustable-optimizer-performance.py: script to compare the optimizer
performance of different methods as a function of the adjustment parameter and/
or correlation between high- and low-fidelity
"""

import argparse
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parse import compile
from pyprojroot import here

import mf2


data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2022-06-28-adjustable-optimizer-performance/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

subfolder_template = compile('{func_name}-{method}-b{init_budget:d}-i{idx:d}')
errorgrid_template = compile('errorgrid_{iteration:d}.nc')

optima = {
    'Branin': mf2.adjustable.branin.high(mf2.adjustable.branin.x_opt),
    'Paciorek': mf2.adjustable.paciorek.high(mf2.adjustable.paciorek.x_opt),
    'Hartmann': mf2.adjustable.hartmann3.high(mf2.adjustable.hartmann3.x_opt),
    'Trid': mf2.adjustable.trid.high(mf2.adjustable.trid.x_opt),
}

folder_template = compile('Adjustable {name} {a:f}-{remainder}')


def main(save_exts=('.png', '.pdf')):
    groups = ['fixed', 'naive']

    for func_name in optima:
        folders = [
            folder
            for folder in data_path.iterdir()
            if func_name in folder.name and folder_template.parse(folder.name)
        ]

        fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

        for group_name in groups:

            group_folders = [f for f in folders if group_name in f.name]

            group = []
            for folder in group_folders:
                df = pd.read_csv(folder / 'log.csv', index_col=0, sep=';')
                y_best = df.loc[df['fidelity'] == 'high']['fitness'].min() - optima[func_name]
                param = folder_template.parse(folder.name)['a']
                group.append((param, float(y_best)))

            group.sort(key=itemgetter(0))
            means = [
                (name, np.mean(np.array(list(values))[:,1]))
                for name, values in groupby(group, key=itemgetter(0))
            ]
            axes.plot(*zip(means), label=f'{group_name} (mean)')
            axes.scatter(*list(zip(*group)), label=f'{group_name} runs', s=10)

        axes.set_ylabel('error')
        axes.set_xlabel('adjustment parameter a')
        axes.legend(loc=1)
        axes.set_yscale('log')
        axes.set_title(func_name)
        for ext in save_exts:
            fig.savefig(plot_path / f'adjustable_{func_name}_fixed_naive_comparison{ext}')
        fig.clear()
        plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exts", action="extend", nargs="+", type=str,
                        help="File extensions to use when saving images. Default: [.PNG, .PDF].")
    args = parser.parse_args()

    kwargs = {}
    if args.exts:
        kwargs['save_exts'] = args.exts

    main(**kwargs)
