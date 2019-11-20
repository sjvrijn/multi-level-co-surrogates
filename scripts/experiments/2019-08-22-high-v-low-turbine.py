#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-08-22-high-v-low-turbine.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples of multiple 
multi-fidelity functions
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess

arguments = range(5)
for arg in arguments:
    subprocess.run(f'NQDIR=turbine nq nice -n 5 python3 2019-08-22-high-v-low.py {arg}', shell=True)

