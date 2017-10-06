#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

'''
config.py: Simple python file to store variable settings that are re-used in multiple files
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


from MultiFidelityFunctions import *
from .local import base_dir


# Filename related settings
plot_dir = base_dir + 'plots/'
data_dir = base_dir + 'data/'
filename = '{dim}D-{func}'
suffix   = '-s{size}-r{rep}-'
plot_ext = '.png'  # Choose from: ['.png', '.pdf']
data_ext = '.dat'

# Dictionary of available functions
fit_funcs = {
    'borehole': borehole,
    'curretal88exp': curretal88exp,
    'park91a': park91a,
    'park91b': park91b,
}

fit_func_dims = {name: len(fit_funcs[name].u_bound) for name in fit_funcs.keys()}

# Experimental parameters
training_size = 50
experiment_repetitions = 30
