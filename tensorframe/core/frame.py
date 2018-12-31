'''
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
'''

from .libs import libio
from .libs import libbinops

from . import parser
from . import arrow


class GPUMemPtr:
    '''Initialise GPU pinned memory'''
    def __init__(self, size='1MB'):
        bytes_size = parser.parse_bytestr(size)
        self.ptr = None
        self.initialise_new(bytes_size)

    def initialise_new(self, size_Mbytes):
        self.ptr = libio.gpu_halloc(size_Mbytes)

    def retrieve_ptr(self):
        return self.ptr


def read_csv(filename, dest, dtype=None):
    pyarrays = libio.read_csv(filename, dest.retrieve_ptr(), arrow.validate_dtype(dtype))
    return arrow.pyarrays_to_pabatch(pyarrays)


def array_sum(pabatch):
    columns = arrow.pabatch_to_tfbatch(pabatch)
    return libbinops.array_sum(columns)


def convert_dtype_from_stringlist(input_list):
    assert(isinstance(input_list, list))
    return [
        arrow.pyarrow_dtypes_from_str(dtype_str)
        for dtype_str in input_list if isinstance(dtype_str, str)
    ]


def generate_csv(outputfile, csv_contents, memory=None):
    if memory == None:
        raise Exception('TODO: allocate memory automatically')

    assert(isinstance(csv_contents, dict))
    assert(isinstance(csv_contents.get('columns'), list))
    assert(isinstance(csv_contents.get('rows_no'), int))

    arrow_datatypes = [
        arrow.pyarrow_dtypes_from_str(column['datatype']) for column in csv_contents.get('columns')
        if isinstance(column.get('datatype'), str)
        and isinstance(column.get('min_value'), int)
        and isinstance(column.get('max_value'), int)
        and isinstance(column.get('name'), str)
    ]

    # assure no items have been skipped
    assert(arrow_datatypes.__len__() == csv_contents.get('columns').__len__())

    # NOTE. generating csv does not currently support writing half floats yet
    # as a workaround we write halffloat as regular float, while still pass the halffloat version for reading
    # which is ok for now because we dont need to actually write a halffloat to test reading one
    import pyarrow as pa
    arrow_datatypes = [pa.float32() if arrow_type == pa.float16() else arrow_type for arrow_type in arrow_datatypes]

    dtypes_dict = arrow.pyarray_dtypes_dict()
    
    def _retrieve_updated_column(no):
        column = csv_contents['columns'][no]
        column['datatype'] = pyarray_type = dtypes_dict[arrow_datatypes[no]]
        return column

    # Convert datatype to a pyarray known value, pass columns that contain all required values
    updated_columns = [_retrieve_updated_column(c) for c in range(0, csv_contents.get('columns').__len__())]

    # Generate
    pyarrays = libio.generate_arrays(memory.retrieve_ptr(), csv_contents.get('rows_no'), updated_columns)
 
    # Convert to pyarrow, and back to tframe (cheap call, schematics-only)
    pabatch = arrow.pyarrays_to_pabatch(pyarrays)
    tframe = arrow.pabatch_to_tf__frame(pabatch)
    
    # TODO: sum it with numpy and store the results for comparison with our own sum implementation

    assert(libio.write_csv(outputfile, tframe) == 0)

