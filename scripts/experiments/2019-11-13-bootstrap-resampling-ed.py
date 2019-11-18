#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-13-bootstrap-resampling-ed.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples using different
surrogate models as backend instead of only using Kriging with Matern Kernel
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess
from itertools import product
import numpy as np

cases = [0, 1, 6, 7]
scales = np.round(np.linspace(0.1, 1.0, 10), 1)
for case_idx, scale in product(cases, scales):
    subprocess.run(f'NQDIR=ed nq python3 2019-11-13-bootstrap-resampling.py {case_idx} {scale}', shell=True)
