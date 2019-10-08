#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
20191007_adjustables_edd.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples of multiple 
adjustable functions
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess

arguments = [1.0, 0,8, 0.6, 0.9, 0.7, 0.95, 0.85, 0.75, 0.65, 0.55]
for arg in arguments:
    subprocess.run(f'NQDIR=edd nq python3 20191007_adjustables.py {arg} > edd.log', shell=True)

