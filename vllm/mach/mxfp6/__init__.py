# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 MXFP6 producer and GDN acceleration paths.

The CUDA operators are provided by the installed ``mxfp6-sm120`` and Mach
extension wheels.  Python dispatch remains capability- and shape-gated so a
stock installation keeps the original vLLM path.
"""
