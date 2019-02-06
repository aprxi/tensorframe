"""
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
"""

import pyarrow as pa

_ENDIANNESS = '<'

_DTYPES_CONV = {
    _ENDIANNESS + 'f2': pa.float16(),
    _ENDIANNESS + 'f4': pa.float32(),
    _ENDIANNESS + 'f8': pa.float64(),
    _ENDIANNESS + 'i2': pa.int16(),
    _ENDIANNESS + 'i4': pa.int32(),
    _ENDIANNESS + 'i8': pa.int64(),
    _ENDIANNESS + 'u2': pa.uint16(),
    _ENDIANNESS + 'u4': pa.uint32(),
    _ENDIANNESS + 'u8': pa.uint64(),
    '|i1': pa.int8(),
    '|u1': pa.uint8(),
}

_DTYPES_CONV_STR = {
    "float16":  pa.float16(),
    "float32":  pa.float32(),
    "float64":  pa.float64(),
    "int16":    pa.int16(),
    "int32":    pa.int32(),
    "int64":    pa.int64(),
    "uint16":   pa.uint16(),
    "uint32":   pa.uint32(),
    "uint64":   pa.uint64(),
    "int8":     pa.int8(),
    "uint8":    pa.uint8(),
}

def _dtypes_from_arrow():
    """Reverse key-value pair of _DTYPES_CONV"""
    return {v: k for k, v in _DTYPES_CONV.items()}
def pyarrow_dtypes_from_str(input_str):
    """Return pyarrow datatype based on string formatted datatype"""
    return _DTYPES_CONV_STR[input_str]
def pyarray_dtypes_from_str(input_str):
    """Return pyarray datatype based on string formatted datatype"""
    return _dtypes_from_arrow()[_DTYPES_CONV_STR[input_str]]
def pyarray_dtypes_dict():
    """Return key-value table of datatypes. Key: arrow datatype. Value: pyarray datatype string"""
    return _dtypes_from_arrow()

def _np_to_array(row_length, arr):
    np_name = arr[0].dtype.descr[0][0]
    np_datatype = arr[0].dtype.descr[0][1]
    return (np_name,
            pa.Array.from_buffers(
                _DTYPES_CONV[np_datatype],
                row_length,
                [pa.py_buffer(memoryview(arr[1])), pa.py_buffer(memoryview(arr[0]))])
           )


def validate_dtype(dtype_in):
    """
    Input is an argument represention one, or more datatypes.

    Per column, number of columns have to match number of columns in csv file:
    dtype = [pa.int32(), pa.int32(), pa.int32(), pa.int32()]
    dtype = {'__columns__': [pa.int32(), pa.int32(), pa.int32(), pa.int32()]}

    Default:
    dtype_in = pa.int32()
    dtype_out = {'__default__': pa.int32()}

    Not yet supported:
    Default, optional column overwrite:
    dtype_in = {'__default__': pa.int32(), '__columns__': {'colname': pa.int32()}}
    dtype_out = raise ValueError

    dtype_in = {'colname': pa.int32()}
    dtype_out = raise ValueError
    """

    if dtype_in is None:
        # use default datatype
        dtype_in = pa.float32()

    argtype = type(dtype_in)
    valid_types = _dtypes_from_arrow()

    if argtype is pa.DataType:
        if dtype_in not in list(valid_types.keys()):
            raise ValueError('Not supporting type: ' + dtype_in.__str__())
        return {'__default__': valid_types[dtype_in]}

    if argtype is dict:
        raise ValueError('Not yet supported dict')

    if argtype is list and dtype_in.__len__() > 0:
        matches = [dtype in list(valid_types.keys()) for dtype in dtype_in]
        if False in matches:
            mismatches = [dtype_in[j].__str__() + '(column:' + str(j) + ')'
                          for j in range(0, len(matches)) if matches[j] is False]
            raise ValueError('List contains unsupported datatype: ' + ','.join(mismatches))

        if set(dtype_in).__len__() == 1:
            # all list members are of same type
            return {'__default__': valid_types[dtype_in[0]]}
        return {'__columns__': list([valid_types[dtype] for dtype in dtype_in])}

    raise ValueError('No input to match datatypes')


def pyarrays_to_pabatch(pyarrays):
    """
    Input a pyarrays list of tuples. Each represents a column of data and contains two items.
    First tuple item is colunm data, second tuple item NaN values (bitstring).
    """
    if not isinstance(pyarrays, list):
        raise ValueError('pyarrays should be a list of array tuple objects')
    if False in list([isinstance(tf_array, tuple) for tf_array in pyarrays]):
        raise ValueError('pyarrays should be a list of array tuple objects')

    # rows are equal. Use first column as reference.
    row_length = pyarrays[0][0].size
    pv_s = [_np_to_array(row_length, tf_array) for tf_array in pyarrays]

    pabatch = pa.RecordBatch.from_arrays([pl[1] for pl in pv_s], [pl[0] for pl in pv_s])
    return pabatch


def pyarrays_to_meta(pyarrays):
    """
    Parse a pyarrays object and return this a tensorframe meta object.
    Tensorframe meta object contains base pyarray information in a more presentable format,
    this can be later supplemented with additional information (e.g. stats information).
    """
    # rows are equal. Use first column as reference.
    row_length = pyarrays[0][0].size

    column_meta = [(
        [memoryview(column[1]), memoryview(column[0])],
        column[0].dtype.descr[0][1],
        column[0].dtype.descr[0][0]
        ) for column in pyarrays if column.__len__() == 2]

    tensorframe_meta = {
        'no_columns': int(pyarrays.__len__()),
        'no_rows': int(row_length),
        'memory_ptr': [meta[0] for meta in column_meta],
        'columns': {
            'names': [meta[2] for meta in column_meta],
            'datatypes': [meta[1] for meta in column_meta],
            'stats':    {},
        }
    }
    return tensorframe_meta


def pabatch_to_tf__frame(pabatch):
    """
    Convert pyarrow recordbatch object to a tensorframe_meta object.
    Note. Function requires an update. pyarrays_to_meta() function is the most recent one.
        - retrieve column_names from pyarrow recordbatch.
        - test on more diverse pyarraw recordbatch objects. It will likely fail if there is more
          than a single memory pointer.
    """
    column_buffers = [column.buffers() for column in pabatch.columns]
    pyarray_types = _dtypes_from_arrow()

    tensorframe_meta = {
        'no_columns': int(pabatch.num_columns),
        'no_rows': int(pabatch.num_rows),
        'memory_ptr': [
            [memoryview(colbuf[0]), memoryview(colbuf[1])]
            for colbuf in column_buffers if colbuf.__len__() == 2
        ],
        'columns': {
            'datatypes': [pyarray_types.get(datatype) for datatype in pabatch.schema.types],
            'stats':    {},
        }
    }
    return tensorframe_meta


def pabatch_to_tfbatch(pabatch):
    """ FUNCTION MARKED FOR OBSOLETION """
    assert pabatch is not None
    raise Exception('Function obsoleted')
    #pyarray_types = _dtypes_from_arrow()

    #columns = [pabatch.column(column_no) for column_no in range(0, pabatch.num_columns)]
    #return [(
    #    memoryview(col.buffers()[-1]),
    #    col.__len__(),
    #    pyarray_types.get(col.type)
    #    ) for col in columns]


def convert_dtype_from_stringlist(input_list):
    """Input a list of string formatted datatypes,
    return a list of matching pyarrow datatype objects."""
    assert isinstance(input_list, list)
    return [
        pyarrow_dtypes_from_str(dtype_str)
        for dtype_str in input_list if isinstance(dtype_str, str)
    ]
