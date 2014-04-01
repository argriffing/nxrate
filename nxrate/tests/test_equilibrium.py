"""
"""
from __future__ import division, print_function, absolute_import

from itertools import combinations

import networkx as nx
import numpy as np
from numpy.testing import assert_raises

import nxrate
from nxrate.util import (
        get_uniform_distn,
        get_random_binom_distn,
        get_random_symmetric_dense_Q,
        get_random_symmetric_sparse_Q,
        get_random_sparse_uniform_distn,
        )
from nxrate.testing import (
        assert_detailed_balance, assert_equilibrium,
        UnweightedDetailedBalanceError, WeightedDetailedBalanceError,
        UnweightedEquilibriumError, WeightedEquilibriumError,
        )


def _get_state_lists():
    return [
            range(2),
            range(3),
            range(4),
            [1, 2, 3],
            ['a', 'b', 'c'],
            ['a', 'b', 'c', 'd'],
            [1, 'x'],
            ]


def _check_ok_random_symmetric_dense_Q(states):
    np.random.seed(1234)
    distn = get_uniform_distn(states)
    Q = get_random_symmetric_dense_Q(states)
    assert_equilibrium(Q, distn)
    assert_detailed_balance(Q, distn)


def _check_ok_random_symmetric_sparse_Q(states):
    np.random.seed(1234)
    distn = get_uniform_distn(states)
    Q = get_random_symmetric_dense_Q(states)
    assert_equilibrium(Q, distn)
    assert_detailed_balance(Q, distn)


def _check_bad_distn_random_symmetric_dense_Q(states):
    np.random.seed(1234)
    distn = get_random_binom_distn(states)
    Q = get_random_symmetric_dense_Q(states)
    assert_raises(WeightedEquilibriumError, assert_equilibrium,
            Q, distn)
    assert_raises(WeightedDetailedBalanceError, assert_detailed_balance,
            Q, distn)


def _check_bad_distn_random_symmetric_sparse_Q(states):
    np.random.seed(1234)
    distn = get_random_binom_distn(states)
    Q = get_random_symmetric_sparse_Q(states)
    assert_raises(WeightedEquilibriumError, assert_equilibrium,
            Q, distn)
    assert_raises(WeightedDetailedBalanceError, assert_detailed_balance,
            Q, distn)


def test_ok_random_symmetric_dense_Q():
    for states in _get_state_lists():
        yield _check_ok_random_symmetric_dense_Q, states


def test_ok_random_symmetric_sparse_Q():
    for states in _get_state_lists():
        if len(set(states)) > 2:
            yield _check_ok_random_symmetric_sparse_Q, states


def test_bad_distn_random_symmetric_dense_Q():
    for states in _get_state_lists():
        yield _check_bad_distn_random_symmetric_dense_Q, states


def test_bad_distn_random_symmetric_sparse_Q():
    for states in _get_state_lists():
        if len(set(states)) > 2:
            yield _check_bad_distn_random_symmetric_sparse_Q, states


def test_equilibrium_but_not_detailed_balance():
    states = list('abcd')
    distn = get_uniform_distn(states)
    Q = nx.DiGraph()
    Q.add_weighted_edges_from([
        ('a', 'b', 2),
        ('b', 'c', 2),
        ('c', 'd', 2),
        ('d', 'a', 2),
        ])
    assert_raises(UnweightedDetailedBalanceError, assert_detailed_balance,
            Q, distn)
    assert_equilibrium(Q, distn)


def test_unweighted_equilibrium_error():
    states = list('abc')
    distn = get_uniform_distn(states)
    Q = nx.DiGraph()
    Q.add_weighted_edges_from([
        ('a', 'b', 2),
        ('b', 'c', 2),
        ])
    assert_raises(UnweightedEquilibriumError, assert_equilibrium,
            Q, distn)
    assert_raises(UnweightedDetailedBalanceError, assert_detailed_balance,
            Q, distn)

