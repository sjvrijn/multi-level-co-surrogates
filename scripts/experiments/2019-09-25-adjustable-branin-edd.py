#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-09-25-adjustable-branin-edd.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess
import numpy as np

arguments = np.round(np.linspace(-0.5, 0.25, 15, endpoint=False), 2)
for arg in arguments:
    subprocess.run(f'NQDIR=edd nq python3 2019-09-25-adjustable-branin.py {arg}', shell=True)

