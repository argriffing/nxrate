"""
Utility functions.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
from itertools import combinations
import random

import networkx as nx
import scipy.stats


def isclose(a, b, rtol=1e-5, atol=1e-8):
    """
    For assertion use rtol=1e-7 atol=0 instead.
    Following numpy, the relative difference (rtol * abs(b)) and
    the absolute difference atol are added together to compare
    against the absolute difference between a and b.
    """
    return abs(a - b) <= (atol + rtol * abs(b))


def get_uniform_distn(states):
    states = set(states)
    nstates = len(states)
    if not nstates:
        raise Exception('the distribution has empty support')
    p = 1 / nstates
    return dict((s, p) for s in states)


def get_random_sparse_uniform_distn(states, nzeros=1):
    """
    States with positive probability have equal probability.
    """
    states = list(set(states))
    random.shuffle(states)
    npositive = len(states) - nzeros
    d = {}
    for i, state in enumerate(states):
        if i < nzeros:
            continue
        else:
            p = 1 / npositive
            d[state] = p
    return d


def get_random_symmetric_dense_Q(states):
    """
    Use a networkx DiGraph even though the rate matrix is dense.
    """
    states = list(set(states))
    random.shuffle(states)
    Q = nx.DiGraph()
    for sa, sb in combinations(states, 2):
        rate = 10 * random.random()
        Q.add_edge(sa, sb, weight=rate)
        Q.add_edge(sb, sa, weight=rate)
    return Q


def get_random_symmetric_sparse_Q(states, nzeros=1):
    states = list(set(states))
    random.shuffle(states)
    Q = nx.DiGraph()
    for i, (sa, sb) in enumerate(combinations(states, 2)):
        if i < nzeros:
            continue
        rate = 10 * random.random()
        Q.add_edge(sa, sb, weight=rate)
        Q.add_edge(sb, sa, weight=rate)
    return Q


def get_random_binom_distn(states, p=0.4):
    states = list(set(states))
    random.shuffle(states)
    n = len(states) - 1
    d = {}
    rv = scipy.stats.binom(n, p)
    for k, state in enumerate(states):
        d[state] = rv.pmf(k)
    return d


def dict_argmin(d):
    if None in d:
        raise Exception('the support should not contain None')
    min_k = None
    min_v = None
    for k, v in d.items():
        if min_k is None or v < min_v:
            min_k = k
            min_v = v
    return min_k


def dict_argmax(d):
    if None in d:
        raise Exception('the support should not contain None')
    max_k = None
    max_v = None
    for k, v in d.items():
        if max_k is None or v > max_v:
            max_k = k
            max_v = v
    return max_k


def nxdistn_to_distn(nxdistn):
    d = {}
    for edge in nxdistn.edges():
        a, b = edge
        d[a, b] = nxdistn[a][b]['weight']
    return d


def get_directed_flow_graph(Q, distn):
    """
    Compute the pairwise directed flows between vertices.

    This is a helper function for checking equilibrium and detailed balance.

    """
    R = nx.DiGraph()
    for sa, sb in Q.edges():
        if sa in distn:
            flow = distn[sa] * Q[sa][sb]['weight']
            R.add_edge(sa, sb, weight=flow)
    return R


def get_marginal_flows(R):
    """
    Compute total flows into and out of vertices.

    Note that this is like weighted networkx DiGraph.in_degree
    and DiGraph.out_degree, but the dicts constructed by this function
    may be sparser because some flows that are zero may not be included.

    """
    flow_in = defaultdict(float)
    flow_out = defaultdict(float)
    for sa, sb in R.edges():
        flow = R[sa][sb]['weight']
        flow_in[sb] += flow
        flow_out[sa] += flow
    return flow_in, flow_out


