"""
"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from numpy.testing import assert_raises

import nxrate
from nxrate.testing import assert_rate_matrix


def test_self_transition():
    Q = nx.DiGraph()
    Q.add_weighted_edges_from([
        ('a', 'b', 2),
        ('b', 'a', 3),
        ('a', 'a', 4),
        ])
    assert_raises(Exception, assert_rate_matrix, Q)


def test_negative_rate():
    Q = nx.DiGraph()
    Q.add_weighted_edges_from([
        ('a', 'b', 2),
        ('b', 'a', -3),
        ])
    assert_raises(Exception, assert_rate_matrix, Q)


def test_unweighted_asymmetric():
    Q = nx.DiGraph()
    Q.add_weighted_edges_from([
        ('a', 'b', 1),
        ])
    assert_rate_matrix(Q)


def test_weighted_asymmetric():
    Q = nx.DiGraph()
    Q.add_weighted_edges_from([
        ('a', 'b', 1),
        ('b', 'a', 2),
        ])
    assert_rate_matrix(Q)


def test_no_edges():
    Q = nx.DiGraph()
    Q.add_node('a')
    assert_rate_matrix(Q)

