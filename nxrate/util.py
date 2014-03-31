"""
Utility functions.

"""
from __future__ import division, print_function, absolute_import


###############################################################################
# Validation of invariants related to stochastic processes.

def assert_stochastic_vector(v):
    if np.any(v < 0) or np.any(1 < v):
        raise Exception(
                'entries of a finite distribution vector should be in '
                'the inclusive interval [0, 1] '
                'min: %s  max: %s' % (min(v), max(v)))
    if not np.allclose(np.sum(v), 1):
        raise Exception(
                'entries of a finite distribution vector should sum to 1')

def assert_rate_matrix(Q):
    if not np.allclose(np.sum(Q, axis=1), 0):
        raise Exception('expected rate matrix rows to sum to zero')
    if np.any(np.diag(Q) > 0):
        raise Exception('expected rate matrix diagonals to be non-positive')
    if np.any(Q - np.diag(np.diag(Q)) < 0):
        raise Exception('expected rate matrix off-diagonals to be non-negative')

def assert_equilibrium(Q, distn):
    if not np.allclose(np.dot(distn, Q), 0):
        raise Exception('the distribution is not at equilibrium')

def assert_detailed_balance(Q, distn):
    S = (Q.T * distn).T
    if not np.allclose(S, S.T):
        raise Exception('the detailed balance equations are not met')
