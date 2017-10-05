#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple

BiFidelityFunction = namedtuple('BiFidelityFunction', ['u_bound', 'l_bound', 'high', 'low'])
TriFidelityFunction = namedtuple('TriFidelityFunction', ['u_bound', 'l_bound', 'high', 'medium', 'low'])
