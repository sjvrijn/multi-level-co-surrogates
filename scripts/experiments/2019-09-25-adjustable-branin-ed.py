#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
20190925_adjustable_branin_ed.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess
import numpy as np

arguments = np.round(np.linspace(0.25, 1.0, 16), 2)
for arg in arguments:
    subprocess.run(f'NQDIR=ed nq python3 2019-09-25-adjustable-branin.py {arg}', shell=True)

