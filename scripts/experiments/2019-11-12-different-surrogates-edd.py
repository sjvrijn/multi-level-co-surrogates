#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-12-different-surrogates-edd.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples using different
surrogate models as backend instead of only using Kriging with Matern Kernel
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess

arguments = [0, 1, 6, 7]
for arg in arguments:
    subprocess.run(f'NQDIR=edd nq python3 2019-11-12-different-surrogates.py {arg}', shell=True)
