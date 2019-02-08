"""
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
"""

from .libs import libbinops

def array_sum(tensorframe_meta):
    """
    Create a numpy ndarray from tensorframe meta data, and apply sum.
    The result, a numpy array of column sums, is to be returned directly.
    """
    return libbinops.array_sum(tensorframe_meta)
