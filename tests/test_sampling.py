from hypothesis import given
from hypothesis.strategies import composite, integers
import numpy as np
from pytest import raises, warns

from pyprojroot import here
import sys
module_path = str(here('scripts/experiments'))
if module_path not in sys.path:
    sys.path.append(module_path)


import multiLevelCoSurrogates as mlcs


def test_split_bifiddoe_errors():
    DoE = mlcs.bi_fidelity_doe(2, 5, 10)
    # invalid num_high
    with raises(ValueError):
        mlcs.split_bi_fidelity_doe(DoE, -1, 7)
    with raises(ValueError):
        mlcs.split_bi_fidelity_doe(DoE, 6, 7)

    # invalid num_low
    with raises(ValueError):
        mlcs.split_bi_fidelity_doe(DoE, 3, 11)

    # # invalid comparison
    with raises(ValueError):
        mlcs.split_bi_fidelity_doe(DoE, 4, 3)


def test_split_bifiddoe_warnings():
    DoE = mlcs.bi_fidelity_doe(2, 5, 10)

    # not enough high-fid
    with warns(mlcs.LowHighFidSamplesWarning):
        mlcs.split_bi_fidelity_doe(DoE, 0, 3)
    with warns(mlcs.LowHighFidSamplesWarning):
        mlcs.split_bi_fidelity_doe(DoE, 1, 3)

    # no high-fid samples in test-set
    with warns(mlcs.NoHighFidTrainSamplesWarning):
        mlcs.split_bi_fidelity_doe(DoE, 5, 7)

    # no non-high-fidelity low-fidelity samples in test set
    with warns(mlcs.NoSpareLowFidSamplesWarning):
        mlcs.split_bi_fidelity_doe(DoE, 3, 3)


def test_split_bifiddoe():
    DoE = mlcs.bi_fidelity_doe(2, 5, 10)

    a_high, a_low = 3, 7
    a, b = mlcs.split_bi_fidelity_doe(DoE, 3, 7)

    assert len(a.low) == a_low
    assert len(a.high) == a_high
    assert len(a.low) + len(b.low) == len(DoE.low)
    assert len(a.high) + len(b.high) == len(DoE.high)
