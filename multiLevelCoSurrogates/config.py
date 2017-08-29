#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

'''
config.py: Simple python file to store a bunch of variable settings that are re-used in multiple files
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


import cma
import numpy as np


# Filename related settings
plot_dir = 'plots/'
data_dir = 'data/'
filename = '{dim}D-{func}'
suffix   = '-s{size}-r{rep}-'  # '-s{size}-r{rep}-'
plot_ext = '.png'  # Choose from: ['.png', '.pdf']
data_ext = '.dat'

# Dictionary of available functions
fit_funcs = {
    'Griewank':   cma.fcts.griewank,
    'Rastrigin':  cma.fcts.rastrigin,
    'Rosenbrock': cma.fcts.rosen,
    'Schaffer':   cma.fcts.schaffer,
    'Schwefel':   cma.fcts.schwefelelli,
    'Sphere':     cma.fcts.sphere,
}

# Experimental parameters
experiment_repetitions = 30
experiment_intervals   = [1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200]
experiment_thresholds  = np.arange(0.5, 8, 0.5)
