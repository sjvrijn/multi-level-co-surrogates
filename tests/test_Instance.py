#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
test_CandidateArchive.py: Set of tests for the mlcs.CandidateArchive
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from pytest import raises, warns
from hypothesis import given, settings
from hypothesis.strategies import lists, integers
from multiLevelCoSurrogates import InstanceSpec


def instance_spec_generator(n=1_000):
    return lists(
        integers(min_value=2, max_value=n),
        min_size=4, max_size=4, unique=True
    )


@given(instance_spec_generator())
@settings(max_examples=50, deadline=500)
def test_instance_spec_length(instance_spec_creation_values):
    a, b, c, d = sorted(instance_spec_creation_values)
    spec = InstanceSpec(min_high=a, min_low=b, max_high=c, max_low=d)

    assert len(spec) == len(list(spec.pixels))


@given(instance_spec_generator(100), integers(min_value=1, max_value=100))
@settings(max_examples=50, deadline=500)
def test_instance_spec_pixels_instance_relation(instance_spec_creation_values, n):
    a, b, c, d = sorted(instance_spec_creation_values)
    spec = InstanceSpec(min_high=a, min_low=b, max_high=c, max_low=d, num_reps=n)

    assert len(list(spec.instances)) == spec.num_reps*len(spec)



def test_instance_spec_invalid_steps():
    with raises(ValueError):
        InstanceSpec(1, 2, step_low=1, step=None)
    with raises(ValueError):
        InstanceSpec(1, 2, step_high=1, step=None)
    with raises(ValueError):
        InstanceSpec(1, 2, step_high=None, step_low=None, step=None)


def test_instance_spec_steps_warning():
    with warns(UserWarning):
        InstanceSpec(1, 2, step=1, step_low=1, step_high=1)


def test_instance_spec_length_errors():
    # steps != 1
    with raises(NotImplementedError):
        len(InstanceSpec(min_high=1, min_low=2, max_high=3, max_low=4, step=2))
    with raises(NotImplementedError):
        len(InstanceSpec(min_high=1, min_low=2, max_high=3, max_low=4, step_low=2))
    with raises(NotImplementedError):
        len(InstanceSpec(min_high=1, min_low=2, max_high=3, max_low=4, step_high=2))

    # max_high !< max_low or min_high !< min_low
    with raises(NotImplementedError):
        len(InstanceSpec(min_high=1, min_low=2, max_high=4, max_low=3, step=2))
    with raises(NotImplementedError):
        len(InstanceSpec(min_high=2, min_low=1, max_high=3, max_low=4, step=2))