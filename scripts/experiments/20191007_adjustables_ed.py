#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
20191007_adjustables_ed.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples of multiple 
adjustable functions
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess

arguments = [0.0, 0.2, 0.4, 0.1, 0.3, 0.5, 0.05, 0.15, 0.25, 0.35, 0.45]
for arg in arguments:
    subprocess.run(f'NQDIR=ed nq python3 20191007_adjustables.py {arg} > ed.log', shell=True)

