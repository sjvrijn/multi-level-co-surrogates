import pytest
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


# split_with_include
def test_split_doe_with_include_high():
    doe = mlcs.bi_fidelity_doe(2, 10, 20)
    X = np.random.rand(1, 2)
    num_high, num_low = 5, 10
    selected, other = mlcs.split_with_include(doe, num_high, num_low, X, 'high')

    assert len(selected.low) == num_low
    assert len(selected.high) == num_high
    assert len(other.low) == len(doe.low) - num_low + 1
    assert len(other.high) == len(doe.high) - num_high + 1

    X = tuple(X[0])
    assert X in set(tuple(x) for x in selected.low)
    assert X in set(tuple(x) for x in selected.high)
    assert X not in set(tuple(x) for x in other.low)
    assert X not in set(tuple(x) for x in other.high)


def test_split_doe_with_include_low():
    doe = mlcs.bi_fidelity_doe(2, 10, 20)
    X = np.random.rand(1, 2)
    num_high, num_low = 5, 10
    selected, other = mlcs.split_with_include(doe, num_high, num_low, X, 'low')

    assert len(selected.low) == num_low
    assert len(selected.high) == num_high
    assert len(other.low) == len(doe.low) - num_low + 1
    assert len(other.high) == len(doe.high) - num_high

    X = tuple(X[0])
    assert X in set(tuple(x) for x in selected.low)
    assert X not in set(tuple(x) for x in selected.high)
    assert X not in set(tuple(x) for x in other.low)
    assert X not in set(tuple(x) for x in other.high)


def test_must_include_no_warning():
    doe = mlcs.bi_fidelity_doe(2, 5, 10)
    X = np.random.rand(1, 2)

    with warns(None) as record:
        mlcs.split_with_include(doe, 4, 5, must_include=X, fidelity='low')
    assert len(record) == 0


# remove_from_bi_fid_doe
def test_remove_from_bifiddoe():
    DoE = mlcs.bi_fidelity_doe(2, 5, 10)
    X = DoE.high[3]
    assert X in DoE.high and X in DoE.low
    DoE = mlcs.remove_from_bi_fid_doe(X=X, DoE=DoE)
    assert X not in DoE.high and X not in DoE.low


# split_bi_fidelity_doe
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


def test_valueerror_split_bifiddoe():
    """This test uses a sample archive that caused a ValueError in the previous
    implementation of split_bi_fid_doe() which used `row in np.ndarray`. The
    reason is that in numpy, this is implemented as `np.any(row == np.ndarray)`,
    which returns True if values match in *AT LEAST ONE* column, instead of all.
    """
    archive_path = here('tests/test-files/split_bi_fid_doe_regression_archive.npy')
    doe = np.load(archive_path, allow_pickle=True).item().as_doe()

    # split_bi_fidelity_doe() would fail under exactly these circumstances:
    np.random.seed(3)
    _ = mlcs.split_bi_fidelity_doe(doe, 11, 12)
