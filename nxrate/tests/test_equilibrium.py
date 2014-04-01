"""
"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from numpy.testing import assert_raises

import nxrate
from nxrate.testing import (assert_equilibrium,
        UnweightedEquilibriumError, WeightedEquilibriumError)
