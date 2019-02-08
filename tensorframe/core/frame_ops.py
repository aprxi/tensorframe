"""
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
"""

from . import pyc_binops
from . import numpy_ops

from . import print_functions as printf

def array_sum(tensorframe_meta, processor='CPU'):
    """
    Sum the columns of a tensorframe object. This job is executed by choice on CPU or GPU.
    The result, a numpy array of column sums, is to be returned directly.
    """

    keyname = 'total_sum'

    if processor == 'GPU':
        printf.warning('SUM FEATURE NOT YET CONSIDERED COMPLETE')
        printf.warning('TO FIX:\n1. SUMS > UINT64.\n2. SUMS OF FLOAT TYPES STILL DONE IN INT.')
        # note on latter: either overflow like numpy, or implement INT128 on CUDA)

        return (keyname, pyc_binops.array_sum(tensorframe_meta))
    if processor == 'CPU':
        return (keyname, numpy_ops.array_sum(tensorframe_meta))
    raise ValueError('processor must either GPU or CPU')
