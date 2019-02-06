"""
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
"""

from .libs import libio
from . import parser
from . import arrow


class GPUMemPtr:
    """Initialise GPU pinned memory"""
    def __init__(self, size='1MB'):
        """Initialise a new memory object."""
        self.ptr = None
        self.initialise_new(parser.parse_bytestr(size, minimum=2**20))

    def initialise_new(self, size_megabytes):
        """Allocated pinned host memory and store the pointer."""
        self.ptr = libio.gpu_halloc(size_megabytes)

    def retrieve_ptr(self):
        """Return pointer to memory object."""
        return self.ptr


def generate_pyarray(csv_contents, memory):
    """
    Generate a (pseudo-) random pyarrays object based on schema, data is stored in memory

    Note:
    Generating csv does not currently support writing half floats yet. Workaround is to write
    halffloat as regular float, while still pass the halffloat version for reading.
    This works for now because we dont need to actually write a halffloat to test reading one.
    """

    assert isinstance(csv_contents, dict)
    assert isinstance(csv_contents.get('columns'), list)
    assert isinstance(csv_contents.get('rows_no'), int)

    arrow_datatypes = [
        arrow.pyarrow_dtypes_from_str(column['datatype']) for column in csv_contents.get('columns')
        if isinstance(column.get('datatype'), str)
        and isinstance(column.get('min_value'), int)
        and isinstance(column.get('max_value'), int)
        and isinstance(column.get('name'), str)
    ]

    # assure no items have been skipped
    assert arrow_datatypes.__len__() == csv_contents.get('columns').__len__()

    import pyarrow as pa
    arrow_datatypes = [
        pa.float32() if arrow_type == pa.float16()
        else arrow_type
        for arrow_type in arrow_datatypes]

    dtypes_dict = arrow.pyarray_dtypes_dict()

    def _retrieve_updated_column(number):
        column = csv_contents['columns'][number]
        column['datatype'] = dtypes_dict[arrow_datatypes[number]]
        return column

    # Convert datatype to a pyarray known value, pass columns that contain all required values
    updated_columns = [
        _retrieve_updated_column(c)
        for c in range(0, csv_contents.get('columns').__len__())]

    # Generate
    return libio.generate_arrays(
        memory.retrieve_ptr(),
        csv_contents.get('rows_no'),
        updated_columns)
