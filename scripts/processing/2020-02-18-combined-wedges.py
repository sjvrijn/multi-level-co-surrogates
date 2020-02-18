#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-02-18-combined-wedges.py: A processing script file to create both the
regular wedge plots and those for the subsampling results at the same time.
By combining the renders, they are guaranteed to use the same colorscale.
"""

import sys
from itertools import product

from pyprojroot import here

import mf2

from experiments import Instance, create_model_error_grid

save_dir = here('files/2020-02-18-combined-wedges/')
save_dir.mkdir(parents=True, exist_ok=True)


