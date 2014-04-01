"""
"""
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_raises

import numpy as np

import nxrate
from nxrate.util import get_uniform_distn, get_random_binom_distn
from nxrate.testing import assert_distn, LocalDistnError, GlobalDistnError


def test_assert_distn_neg_element():
    d = {
            'a' : 0,
            'b' : -0.1,
            'c' : 1.1}
    assert_raises(LocalDistnError, assert_distn, d)


def test_assert_distn_large_element():
    d = {
            'a' : 0,
            'b' : 2,
            'c' : 0.5}
    assert_raises(LocalDistnError, assert_distn, d)


def test_assert_distn_large_sum():
    d = {
            'a' : 0.5,
            'b' : 0.5,
            'c' : 0.5}
    assert_raises(GlobalDistnError, assert_distn, d)


def test_assert_distn_ok_1():
    d = {'a' : 1}
    assert_distn(d)


def test_assert_distn_ok_uniform_2():
    d = {'a' : 0.5, 'b' : 0.5}
    assert_distn(d)


def test_assert_distn_ok_uniform_4():
    d = {
            'a' : 0.25, 'b' : 0.25,
            'c' : 0.25, 'd' : 0.25}
    assert_distn(d)


def test_assert_distn_ok_nonuniform_2():
    d = {'a' : 0.25, 'b' : 0.75}
    assert_distn(d)


def test_uniform_distn():
    for n in range(1, 5):
        states = range(n)
        d = get_uniform_distn(states)
        assert_distn(d)


def test_random_binom_distn():
    np.random.seed(1234)
    for n in range(1, 5):
        states = range(n)
        d = get_random_binom_distn(states)
        assert_distn(d)

