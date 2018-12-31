/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include "libbinops.h"
#include "tf_types.h"

#include <string>
#include <sstream>

PyObject* pyobj__array_sum(PyObject* self, PyObject* args){
// TODO: function does not yet contain proper eror checking.
    uint32_t ii;
    uint32_t no_columns;

    char *char_ptr;
    uint8_t datatype;
    uint64_t no_items;
    void *memory_ptr;
    void *dest;
    
    int64_t *p_long;
    double *p_double;
    uint8_t dtype;

    uint8_t stride_dest = 8;

    Py_buffer *memory_buf;
    PyObject *array_list;
    PyObject *array_item;
    PyObject *tuple_item;

    // Parse and validate input
    if(!PyArg_ParseTuple(args, "O", &array_list) || PyList_Check(array_list) != true){
        PyErr_SetString(PyExc_ValueError, "pyobj__array_sum() requires list as an input.");
        return(NULL);
    }
    if((uint64_t) PyList_Size(array_list) > UINT32_MAX){
        PyErr_SetString(
            PyExc_ValueError,
            ("pyobj__array_sum(). Input list exceeded max number of arrays. Max:" + std::to_string(UINT32_MAX)).c_str()
        );
        return(NULL);
    }
    if((no_columns = (uint32_t) PyList_Size(array_list)) == 0){
        Py_INCREF(Py_None);
        return Py_None;
    }

    // Allocate a data structure to hold column information
    tf_column_obj *columns = (tf_column_obj*) malloc(no_columns * sizeof(tf_column_obj));
    if(!columns){ return PyErr_NoMemory();};

    // Parse python array list into a column obj structure
    for(ii = 0; ii < no_columns; ii++){
        // List items should be tuples of format: (ptr, size, datatype)
        if(!(array_item = PyList_GetItem(array_list, ii))
            || PyTuple_Check(array_item) != true
            || PyTuple_Size(array_item) != 3
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__array_sum(). Error parsing item from list, list_no: " + std::to_string(ii)).c_str()
            );
            free(columns);
            return(NULL);
        }

        // Retrieve memory point object
        if(!(tuple_item = PyTuple_GetItem(array_item, 0))
            || PyMemoryView_Check(tuple_item) != true
            || !(memory_buf = PyMemoryView_GET_BUFFER(tuple_item))
            || !(memory_ptr = (void*) memory_buf->buf)
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__array_sum(). Error parsing tuple(0, *ptr), list_no: " + std::to_string(ii)).c_str()
            );
            free(columns);
            return(NULL);
        }

        // Retrieve number of items
        if(!(tuple_item = PyTuple_GetItem(array_item, 1))
            || PyLong_Check(tuple_item) != true
            || sizeof(PyLong_AsLong(tuple_item)) != sizeof(uint64_t)
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__array_sum(). Error parsing tuple(1, no_items), list_no: " + std::to_string(ii)).c_str()
            );
            free(columns);
            return(NULL);
        }
        no_items = (uint64_t) PyLong_AsLong(tuple_item);

        // Retrieve datatype
        if(!(tuple_item = PyTuple_GetItem(array_item, 2))
            || PyUnicode_Check(tuple_item) != true
            || PyUnicode_GET_LENGTH(tuple_item) > 4
            || !(char_ptr=(char*)PyUnicode_DATA(tuple_item))
            || _datatype_column_typebyname(char_ptr, &datatype) != true
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__array_sum(). Error parsing tuple(2, datatype), list_no: " + std::to_string(ii)).c_str()
            );
            free(columns);
            return(NULL);
        }

        // Write to column struct
        columns[ii].memory_ptr = memory_ptr;
        columns[ii].no_items = no_items;
        columns[ii].datatype = datatype;
    }

    if(!(dest = (void*) malloc(no_columns * stride_dest))){ PyErr_NoMemory();};
  
    cu__array_sum(columns, dest, stride_dest, no_columns, no_items);
    free(columns);

    // NOTE: current implementation returns column totals to python as a list of pylong objects
    // within python it can then be then repacked into the datatype of choice.
    // TODO: figure out a more generic approach that is also capable of handling bigger sum numbers (>8byte types) if needed.

    array_list = PyList_New((Py_ssize_t) no_columns);
 
    p_long = (int64_t*) dest;
    p_double = (double*) dest;

    for(ii = 0; ii < no_columns; ii++){
        dtype = columns[ii].datatype;

        if(dtype >= TF_INT8 && dtype <= TF_UINT64){
            PyList_SetItem(array_list, ii, PyLong_FromLong(p_long[ii]));
        }
        else if(dtype >= TF_FP16 && dtype <= TF_FP64){
            PyList_SetItem(array_list, ii, PyLong_FromDouble(p_double[ii]));
        }
        else{ 
            // default to 0
            PyList_SetItem(array_list, ii, PyLong_FromLong(0));
        }
    }

    return array_list;  //Py_None;
};


PyMODINIT_FUNC
PyInit_libbinops(void)
{
    import_array();
    return PyModule_Create(&cModPyDem_libbinops);
}

