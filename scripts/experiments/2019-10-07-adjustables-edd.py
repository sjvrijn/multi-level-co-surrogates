#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-10-07-adjustables-edd.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples of multiple 
adjustable functions
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess
import numpy as np

arguments = np.round(np.linspace(0.5, 1.0, 11), 2)
for arg in arguments:
    subprocess.run(f'NQDIR=edd nq python3 20191007_adjustables.py {arg}', shell=True)

