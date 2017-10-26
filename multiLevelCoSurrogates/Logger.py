#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

'''
Logger.py: A logging class to easily and automatically write fitness results to a logfile
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


class Logger:
    """A generic logging class to store fitness values as they are created for later post-processing"""

    def __init__(self, fname, header=None):
        self.fname = fname
        self.count = 1
        if header is not None:
            with open(self.fname, 'w', encoding='utf8') as f:
                f.write(header)
                f.write('\n')

    def writeLine(self, data):
        with open(self.fname, 'a', encoding='utf8') as f:
            f.write(str(self.count))
            f.write(' ')
            f.write(' '.join(str(d) for d in data))
            f.write('\n')
        self.count += 1

    def writeLines(self, data):
        for dat in data:
            self.writeLine(dat)
