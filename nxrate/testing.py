"""
Functions for assertions and testing.

In this module, each state is hashable.

Glossary of naming conventions in this module.
    distn : finite distribution over keys of a Python dict
    nxdistn : finite distribution over edges of an nx.DiGraph
    Q : rate matrix as an nx.DiGraph
    R : flow graph as an nx.DiGraph

"""
from __future__ import division, print_function, absolute_import

import random
import scipy.stats

from .util import (dict_argmin, dict_argmax,
        nxdistn_to_distn, isclose,
        get_directed_flow_graph, get_marginal_flows)

__all__ = [
        'assert_distn', 'assert_nxdistn',
        'assert_equilibrium', 'assert_detailed_balance',
        ]

class DistnError(Exception): pass
class LocalDistnError(DistnError): pass
class GlobalDistnError(DistnError): pass

class EquilibriumError(Exception): pass
class UnweightedEquilibriumError(EquilibriumError): pass
class WeightedEquilibriumError(EquilibriumError): pass

class DetailedBalanceError(Exception): pass
class UnweightedDetailedBalanceError(DetailedBalanceError): pass
class WeightedDetailedBalanceError(DetailedBalanceError): pass


def assert_distn(d):
    """

    Parameters
    ----------
    d : dict
        finite distribution

    """
    # Check for probabilities that are too small.
    min_k = dict_argmin(d)
    min_v = d[min_k]
    if min_v < 0:
        raise LocalDistnError('probabilities must be non-negative, '
                'but found prob(%s) : %f' % (min_k, min_v))

    # Check for probabilities that are too large.
    max_k = dict_argmax(d)
    max_v = d[max_k]
    if max_v > 1:
        raise LocalDistnError('probabilities must not be greater than 1, '
                'but found prob(%s) : %f' % (max_k, max_v))

    # Check that the sum of probabilities is nearly 1.
    total = sum(d.values())
    if not isclose(total, 1):
        raise GlobalDistnError('probabilities should add up to 1, '
                'but found total : %f' % total)


def assert_nxdistn(nxdistn):
    assert_distn(nxdistn_to_distn(nxdistn))


def assert_rate_matrix(Q):
    for sa, sb in Q.edges():
        if sa == sb:
            raise Exception('self-transitions are not allowed '
                    'in this networkx representation of rate matrices')
        rate = Q[sa][sb]['weight']
        if rate < 0:
            raise Exception('negative rate '
                    'from %s to %s: %f' % (sa, sb, rate))


def assert_equilibrium(Q, distn, check_inputs=True):
    """
    Assert that the net flow out of each state is near zero.

    """
    # Check the inputs.
    if check_inputs:
        assert_rate_matrix(Q)
        assert_distn(distn)

    # Compute pairwise flows between vertices.
    R = get_directed_flow_graph(Q, distn)

    # Compute the flow into and out of each state.
    flow_in, flow_out = get_marginal_flows(R)

    # Check that each state with flow out also has flow in.
    imba = set(flow_out) - set(flow_in)
    if imba:
        raise UnweightedEquilibriumError('the following states have flow out '
                'but not in: %s' % str(imba))

    # Check that each state with flow in also has flow out.
    imba = set(flow_in) - set(flow_out)
    if imba:
        raise UnweightedEqulibriumError('the following states have flow in '
                'but not out: %s' % str(imba))

    # Check that the net flow out of each vertex is negligible.
    states = set(flow_in) & set(flow_out)
    for s in states:
        if not isclose(flow_in[s], flow_out[s]):
            raise WeightedEquilibriumError('equilibrium fails for state %s: '
                    'flow in: %f  flow out: %f' % (
                        s, flow_in[s], flow_out[s]))


def assert_detailed_balance(Q, distn, check_inputs=True):
    # Check the inputs.
    if check_inputs:
        assert_rate_matrix(Q)
        assert_distn(distn)

    # Compute pairwise flows between vertices.
    R = get_directed_flow_graph(Q, distn)

    # Check that flow existence is symmetric between vertex pairs.
    edges_ab = set(R.edges())
    implied_edges_ba = set((b, a) for a, b in edges_ab)
    imba = edges_ab - implied_edges_ba
    if imba:
        raise UnweightedDetailedBalanceError('detailed balance fails '
                'because only the forward direction of flow exists '
                'for the following state pairs: %s' % str(imba))

    # Check that flow quantity is symmetric between vertex pairs.
    for sa, sb in edges_ab:
        flow_ab = R[sa][sb]['weight']
        flow_ba = R[sb][sa]['weight']
        if not isclose(flow_ab, flow_ba):
            raise WeightedDetailedBalanceError('detailed balance fails '
                    'for state pair (%s, %s): '
                    'forward flow: %f  backward flow: %f' % (
                        sa, sb, flow_ab, flow_ba))

