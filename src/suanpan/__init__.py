# SPDX-FileCopyrightText: 2025 Stefano Miccoli <stefano.miccoli@polimi.it>
#
# SPDX-License-Identifier: MIT

"""Python interface to Abaqus .fil files"""

from ._version import __version__


MAX_ABAQUS_UNSIGNED = 999_999_999
MAX_NODE_NUMBER = MAX_ABAQUS_UNSIGNED
MAX_ELEMENT_NUMBER = MAX_ABAQUS_UNSIGNED
