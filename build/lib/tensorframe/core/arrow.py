'''
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
'''

import pyarrow as pa

_ENDIANNESS = '<'

_dtypes_conv = {
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

_dtypes_conv_str = {
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
    return {v: k for k, v in _dtypes_conv.items()}

# TODO: while not a bottleneck for current use, we need to implement a more efficient approach.
# these functions are used alot inline and we dont need to keep converting static items.
def pyarrow_dtypes_from_str(input_str):
    return _dtypes_conv_str[input_str]
def pyarray_dtypes_from_str(input_str):
    arrow_dt = _dtypes_conv_str[input_str]
    return _dtypes_from_arrow()[arrow_dt]
def pyarray_dtypes_dict():
    return _dtypes_from_arrow()

def _np_to_array(row_length, arr):
    np_name = arr[0].dtype.descr[0][0]
    np_datatype = arr[0].dtype.descr[0][1]
    return (np_name, 
            pa.Array.from_buffers(
                _dtypes_conv[np_datatype],
                row_length,
                #[None,  pa.py_buffer(memoryview(arr[0]))])
                [pa.py_buffer(memoryview(arr[1])),  pa.py_buffer(memoryview(arr[0]))])
            )


def validate_dtype(dtype_in):
    '''
    Per column. Number of columns have to match number of columns in csv file
    dtype = [pa.int32(), pa.int32(), pa.int32(), pa.int32()]
    dtype = {'__columns__': [pa.int32(), pa.int32(), pa.int32(), pa.int32()]}

    Default
    dtype = pa.int32()
    dtype = {'__default__': pa.int32()}

    Default, with optional column overwrite
    dtype = {'__default__': pa.int32(), '__columns__': {'colname': pa.int32()}}
    //dtype = {'colname': pa.int32()}
    '''
   
    if dtype_in is None:
        raise ValueError('No input to match datatypes')
    
    argtype = type(dtype_in)
    valid_types = _dtypes_from_arrow()

    if argtype is pa.DataType:
        if dtype_in not in list(valid_types.keys()):
            raise ValueError('Not supporting type: ' + dtype_in.__str__())
        return {'__default__': valid_types[dtype_in]}

    if argtype is dict:
        print(dtype_in)
        raise ValueError('Not yet supported dict')
    
    if argtype is list and dtype_in.__len__() > 0:
        matches = [dtype in list(valid_types.keys()) for dtype in dtype_in]
        if False in matches:
            mismatches = [dtype_in[j].__str__() + '(column:' + str(j) + ')'
                         for j in range(0, len(matches)) if matches[j] == False]
            raise ValueError('List contains unsupported datatype: ' + ','.join(mismatches))

        if set(dtype_in).__len__() == 1:
            #All list members are of same type
            return {'__default__': valid_types[dtype_in[0]]}
        else:
            return {'__columns__': list([valid_types[dtype] for dtype in dtype_in])}

    raise ValueError('No input to match datatypes')


def pyarrays_to_pabatch(pyarrays):
    '''
    Input a list of tuple items. Each tuple is a column, consisting of column data (0) and na value data (1).
    '''
    if not isinstance(pyarrays, list):
        raise ValueError('pyarrays should be a list of array tuple objects')
    if False in list([isinstance(tf_array, tuple) for tf_array in pyarrays]):
        raise ValueError('pyarrays should be a list of array tuple objects')

    row_length = pyarrays[0][0].size    #Rows are equal. Use first column as reference.
    pv_s = [_np_to_array(row_length, tf_array) for tf_array in pyarrays]

    pabatch = pa.RecordBatch.from_arrays([pl[1] for pl in pv_s], [pl[0] for pl in pv_s])
    return pabatch


def pabatch_to_tf__frame(pabatch):
    '''
    Convert pyarrow batch to tf__frame
    Note. In its current state this is only guaranteed to work for our internal testing case
    TODO: rewrite that it can accept any pyarrow batch object
    '''
    column_buffers = [column.buffers() for column in pabatch.columns]
    pyarray_types = _dtypes_from_arrow()

    frame = {
        'no_columns': int(pabatch.num_columns),
        'no_rows': int(pabatch.num_rows),
        'memory_ptr': [
            [memoryview(colbuf[0]), memoryview(colbuf[1])]
            for colbuf in column_buffers if colbuf.__len__() == 2
        ],
        'datatypes': [pyarray_types.get(datatype) for datatype in pabatch.schema.types]
    }
    return frame


def pabatch_to_tfbatch(pabatch):
    pyarray_types = _dtypes_from_arrow()

    columns = [pabatch.column(column_no) for column_no in range(0, pabatch.num_columns)]
    return [(memoryview(col.buffers()[-1]), col.__len__(), pyarray_types.get(col.type)) for col in columns]
