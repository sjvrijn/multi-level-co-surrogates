#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-01-extend-ratios-turbine.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples of multiple 
multi-fidelity functions
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess

arguments = range(1, 15)
for arg in arguments:
    subprocess.run(f'NQDIR=turbine nq nice -n 5 python3 2019-11-01-extend-ratios.py {arg}', shell=True)

