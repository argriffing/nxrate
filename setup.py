#!/usr/bin/env python
"""Functions related to CTMP rate matrices represented by networkx DiGraphs.

"""

DOCLINES = __doc__.split('\n')

# This setup script is written according to
# http://docs.python.org/2/distutils/setupscript.html
#
# It is meant to be installed through github using pip.

from distutils.core import setup

# This idiom is used by scipy to check if it is running during the setup.
__NXMCTREE_SETUP__ = True

setup(
        name='nxrate',
        version='0.1',
        description=DOCLINES[0],
        author='alex',
        url='https://github.com/argriffing/nxrate/',
        download_url='https://github.com/argriffing/nxrate/',
        packages=['nxrate'],
        test_suite='nose.collector',
        package_data={'nxrate' : ['tests/test_*.py']},
        )


