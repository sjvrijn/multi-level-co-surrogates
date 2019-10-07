#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
20190925_adjustable_branin_ed.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess

arguments = [.3, .4, .6, .7, .8, .9, 1.0]
for arg in arguments:
    subprocess.run(f'NQDIR=edd nq python3 20190925_adjustable_branin.py {arg} > edd.log', shell=True)

