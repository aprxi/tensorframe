"""
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
"""

from . import pyc_io
from . import frame_ops
from . import arrow
from . import output


class PyArrayFrame:
    """Base class of pure pyarray object and core functionality (I/O, base Binary ops).
    Expose low level data API, and zero-copy functions to/ from numpy and pyarrow."""
    def __init__(self, memory):
        self.memory = None
        self.pyarrays = None
        self.frame_meta = None

        if memory is not None:
            self.memory = pyc_io.GPUMemPtr(size=memory)

    def __init_frame_meta(self):
        self.frame_meta = arrow.pyarrays_to_meta(self.pyarrays)

    def __update_stat(self, stat_function, processor='GPU'):
        (keyname, sum_array) = stat_function(self.frame_meta, processor)
        self.frame_meta['columns']['stats'][keyname] = sum_array
        return sum_array

    def sum(self):
        """Calculate the sum of each array, and return it as a list."""
        column_totals = self.frame_meta['columns']['stats'].get('total_sum')
        if column_totals is None:
            #self._PyArrayFrame__update_stat(frame_ops.array_sum, processor='CPU')
            self.__update_stat(frame_ops.array_sum, processor='CPU')
            column_totals = self.frame_meta['columns']['stats'].get('total_sum')
        return column_totals

    def read_csv(self, filename, dtype=None):
        """Load a .csv formatted file."""
        self.pyarrays = pyc_io.libio.read_csv(
            filename,
            self.memory.retrieve_ptr(),
            arrow.validate_dtype(dtype))
        self.__init_frame_meta()
        return self

    def write_csv(self, filename, write_meta=False):
        """Export data contents to a .csv formatted file."""
        if pyc_io.libio.write_csv(filename, self.frame_meta) != 0:
            return False
        if write_meta:
            return output.write_json_serialized(filename + '.meta', self.frame_meta.get('columns'))
        return True

    def random(self, column_schema):
        """(Re-)initialise with (pseudo-) random data."""
        self.pyarrays = pyc_io.generate_pyarray(column_schema, self.memory)
        self.__init_frame_meta()
        return self

    def to_arrow(self):
        """Return a pyarrow recordbatch object. This is a zero-copy operation."""
        return arrow.pyarrays_to_pabatch(self.pyarrays)


class TensorFrame(PyArrayFrame):
    """TensorFrame Object builds on PyArrayFrame. Methods to interface with external programs are
    added here."""
    def __init__(self, memory=None):
        super().__init__(memory)

    def meta(self):
        """
        Return meta information on TensorFrame class in dictionary format.
        """
        return self.frame_meta
