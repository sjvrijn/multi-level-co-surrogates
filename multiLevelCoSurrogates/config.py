#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

'''
config.py: Simple python file to store variable settings that are re-used in multiple files
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


from MultiFidelityFunctions import *
from collections import namedtuple


# Filename related settings
plot_dir = 'plots/'
data_dir = 'data/'
filename = '{dim}D-{func}'
suffix   = '-s{size}-r{rep}-'
plot_ext = '.png'  # Choose from: ['.png', '.pdf']
data_ext = '.dat'

MultiFidelityFunction = namedtuple('MultiFidelityFunction', ['high', 'low'])
# Dictionary of available functions
fit_funcs = {
    'borehole': MultiFidelityFunction(borehole, borehole_lf),
    'curretal88exp': MultiFidelityFunction(curretal88exp, curretal88exp_lf),
    'park91a': MultiFidelityFunction(park91a, park91a_lf),
    'park91b': MultiFidelityFunction(park91b, park91b_lf)
}

# Experimental parameters
training_size = 50
experiment_repetitions = 30
