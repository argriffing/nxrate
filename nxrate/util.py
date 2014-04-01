"""
Utility functions.

"""
from __future__ import division, print_function, absolute_import


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

