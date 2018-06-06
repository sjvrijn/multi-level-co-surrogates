#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
config.py: Simple python file to store variable settings that are re-used in multiple files
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


from MultiFidelityFunctions import *
from .local import base_dir


# Filename related settings
plot_dir = base_dir + 'plots/'
data_dir = base_dir + 'data/'
folder_name = '{ndim}D/{func}/{use}-{surr}/'
suffix   = 's{size}-r{rep}-g{gen}-'
plot_ext = 'png'  # Choose from: ['png', 'pdf']
data_ext = 'dat'

# Dictionary of available functions
fit_funcs = {
    # 'oneDimensional': oneDimensional,
    'bohachevsky': bohachevsky,
    'branin': branin,
    'booth': booth,
    'himmelblau': himmelblau,
    'sixHumpCamelBack': sixHumpCamelBack,
    # 'curretal88exp': curretal88exp,
    'park91a': park91a,
    'park91b': park91b,
    'borehole': borehole,
}

fit_func_dims = {name: len(fit_funcs[name].u_bound) for name in fit_funcs.keys()}

# Experimental parameters
training_sizes = [0]  # [0, 10, 25, 50, 75, 100]  # '0' will retrain on all data
experiment_repetitions = 10
