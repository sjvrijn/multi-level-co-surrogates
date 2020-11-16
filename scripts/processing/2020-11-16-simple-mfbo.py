#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-11-16-simple-mfbo.py: Some initial plotting of data gathered in .csv/.pkl
files during the simple-mfbo runs
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mf2
from pyprojroot import here

data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2020-09-28-shape-reduction/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

# ## Plotting progress of $\tau$ over time

for func in mf2.bi_fidelity_functions:
    tracking_file = data_path.joinpath(f'{func.name}-tracking.csv')
    if tracking_file.exists():
        print(func.name)
        df = pd.read_csv(tracking_file, index_col=0)
        df['tau'].plot()
        plt.savefig(f'{func.name}-tau.pdf', bbox_inches='tight')
        plt.show()
        plt.clf()

# ## Plotting best high/low-fidelity evaluation over time
# ### New-style: using `df[fitness]`

for func in mf2.bi_fidelity_functions:
    tracking_file = data_path.joinpath(f'{func.name}-tracking.csv')
    if tracking_file.exists():
        print(func.name)
        df = pd.read_csv(tracking_file, index_col=0)
        for name, sub_df in df.groupby('fidelity'):
            time = sub_df['budget'].values
            fitnesses = sub_df['fitness'].values
            min_fit = np.minimum.accumulate(fitnesses)

            plt.plot(time, fitnesses, label=f'{name}-fidelity over time')
            plt.plot(time, min_fit, label=f'best {name}-fidelity over time')
            plt.legend(loc=0)
            plt.show()



# ### Old-style: using `archive`

# from numpy import array, nan
#
# with open(data_path / 'Branin-archive.pkl', 'rb') as f:
#     archive_str = load(f)
# archive = eval(archive_str)
# print(archive)
#
#
# low_fid_evals = [values[1] for values in archive.values()]
# min_low_over_time = np.minimum.accumulate(low_fid_evals)
# plt.plot(low_fid_evals, label='low-fidelity evaluations')
# plt.plot(min_low_over_time, label='minimum over time')
# plt.legend(loc=0)
# plt.show()
#
#
# high_fid_evals = [values[0] for values in archive.values()]
# min_high_over_time = np.minimum.accumulate(high_fid_evals)
# plt.plot(high_fid_evals, label='high-fidelity evaluations')
# plt.plot(min_high_over_time, label='minimum over time')
# plt.legend(loc=0)
# plt.show()





