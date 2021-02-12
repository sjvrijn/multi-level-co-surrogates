# -*- coding: utf-8 -*-

"""
2020-02-25-illustrating-1dforrester.py: runner file to run a few
examples of the 1D forrester function to investigate why there is
a jump in MSE when n_low / n_high == 1.5.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from itertools import product
import sys

from pyprojroot import here

import mf2

from experiments import Instance, create_model_error_grid

save_dir = here('files/2020-02-25-illustrating-1dforrester/')
save_dir.mkdir(parents=True, exist_ok=True)

cases = [
    mf2.forrester,
]

kernels = ['Matern']
scaling_options = [
    'off',
]

min_high, max_high = 10, 30
ratio = 1.5
instances = [Instance(h, l, 1)
             for h in range(min_high, max_high+1)
             for l in range(int(h*ratio)-1, int(h*ratio)+2)]

full_cases = list(product(cases, kernels, scaling_options))
extra_attributes = {'mf2_version': mf2.__version__}


for case, kernel, scale in full_cases:
    mfbo_options = {'kernel': kernel, 'scaling': scale}
    create_model_error_grid(case, instances, mfbo_options, save_dir=save_dir,
                            plot_1d=True, extra_attributes=extra_attributes)
