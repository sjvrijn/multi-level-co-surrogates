#!/usr/bin/python
# -*- coding: utf-8 -*-

from MultiFidelityFunctions import BiFidelityFunction

"""
SC.py:
Six-Hump Camel-Back function

Authors: Sander van Rijn, Leiden University


THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
derivative works, such modified software should be clearly marked.
Additionally, this program is free software you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation version 2.0 of the License.
Accordingly, this program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.
"""


def sixHumpCamelBack_hf(xx):
    """
    SIX-HUMP CAMEL-BACK FUNCTION

    INPUT:
    xx = [x1, x2]
    """
    x1, x2 = xx

    term1 = 4*x1**2 - 2.1*x1**4 + x1**6/3
    term2 = x1*x2
    term3 = 4*x2**2 + 4*x2**4

    return term1 + term2 + term3


def sixHumpCamelBack_lf(xx):
    """
    SIX-HUMP CAMEL-BACK FUNCTION, LOWER FIDELITY CODE
    Calls: sixHumpCamelBack_hf
    This function, from Dong et al. (2015), is used as the "low-accuracy code"
    version of the function sixHumpCamelBack_hf.

    INPUT:
    xx = [x1, x2]
    """
    x1, x2 = xx

    term1 = sixHumpCamelBack_hf([0.7*x1, 0.7*x2])
    term2 = x1*x2 - 15

    return term1 + term2


l_bound = [-3, -3]
u_bound = [ 3,  3]

sixHumpCamelBack = BiFidelityFunction(u_bound, l_bound, sixHumpCamelBack_hf, sixHumpCamelBack_lf)
