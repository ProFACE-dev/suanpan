# SPDX-FileCopyrightText: 2025 Stefano Miccoli <stefano.miccoli@polimi.it>
#
# SPDX-License-Identifier: MIT
"""
CLI commands: installed with optional dependency 'cli'.
"""

from importlib.util import find_spec

if find_spec("click") is None or find_spec("h5py") is None:
    msg = "CLI commands require optional dependency 'cli'"
    raise SystemExit(msg)
