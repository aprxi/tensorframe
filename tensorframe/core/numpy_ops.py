"""
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
"""

import numpy as np


def array_sum(tensorframe_meta):
    """
    Create a numpy ndarray from tensorframe meta data, and apply sum.
    The result, a numpy array of column sums, is to be returned directly.
    """
    return [np.ndarray(
        shape=(tensorframe_meta['no_rows'],),
        buffer=tensorframe_meta['memory_ptr'][n][1],
        dtype=tensorframe_meta['columns']['datatypes'][n]).sum()
            for n in range(0, tensorframe_meta['no_columns'])]
