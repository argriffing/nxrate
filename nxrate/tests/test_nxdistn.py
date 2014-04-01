"""
"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from numpy.testing import assert_raises

import nxrate
from nxrate.testing import assert_nxdistn, LocalDistnError, GlobalDistnError


def test_nxdistn_one_edge():
    nxdistn = nx.DiGraph()
    nxdistn.add_weighted_edges_from([
        ('a', 'b', 1),
        ])
    assert_nxdistn(nxdistn)


def test_nxdistn_two_edges():
    nxdistn = nx.DiGraph()
    nxdistn.add_weighted_edges_from([
        ('a', 'b', 0.4),
        ('b', 'a', 0.6),
        ])
    assert_nxdistn(nxdistn)


def test_nxdistn_neg_element():
    nxdistn = nx.DiGraph()
    nxdistn.add_weighted_edges_from([
        ('a', 'b', -0.5),
        ])
    assert_raises(LocalDistnError, assert_nxdistn, nxdistn)


def test_nxdistn_large_element():
    nxdistn = nx.DiGraph()
    nxdistn.add_weighted_edges_from([
        ('a', 'b', 1.5),
        ])
    assert_raises(LocalDistnError, assert_nxdistn, nxdistn)


def test_nxdistn_large_sum():
    nxdistn = nx.DiGraph()
    nxdistn.add_weighted_edges_from([
        ('a', 'b', 0.4),
        ('b', 'c', 0.5),
        ('c', 'a', 0.6),
        ])
    assert_raises(GlobalDistnError, assert_nxdistn, nxdistn)


