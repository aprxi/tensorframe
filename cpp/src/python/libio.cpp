/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include "libio.h"
#include "tf_types.h"

#include <string>
#include <sstream>

using namespace std;

PyObject* pyobj__gpu_halloc(PyObject* self, PyObject* args){
// TODO: function does not yet contain proper eror checking.
    PyObject *tmp;
    char *string;
    
    if(!(tmp=PyTuple_GetItem(args, 0)) || PyLong_Check(tmp) != true){
        PyErr_SetString(PyExc_ValueError, "pyobj__gpu_halloc(), input arguments needs to be a pylong");
        return(NULL);
    }
    if(PyLong_AsLong(tmp) < (1024*1024)){
        //TODO: set this through defined
        PyErr_SetString(PyExc_ValueError, "pyobj__gpu_halloc(), value must me bigger than 1MB");
        return(NULL);
    }
    cu_malloc_carr(&string, (unsigned long long) PyLong_AsLong(tmp));
    return(PyCapsule_New(string, "hostmemory_ptr", NULL));
}


PyObject* pyobj__read_csv(PyObject* self, PyObject* args){
// TODO: function does not yet contain proper eror checking.
    char *arg_str = NULL;
    char *tmp = NULL;
    
    void *dest;
    
    struct file_object fobj;
    uint64_t no_columns;
    uint64_t j_col;   
    uint8_t *columns;
    uint8_t *column_default;

    column_default = (uint8_t*) malloc(sizeof(uint8_t));

    PyObject_Print(args, stdout, 0);

    PyObject *capsule_mem;

    PyObject *dtype_dict;
    PyObject *dtype_dict_obj;
    PyObject *dtype_tmp;

    if(!PyArg_ParseTuple(args, "sOO", &arg_str, &capsule_mem, &dtype_dict)){
        PyErr_SetString(PyExc_ValueError, "pyobj__read_csv(), arguments require (str, memory obj, dict)");
        return(NULL);
    }

    if(PyDict_Check(dtype_dict) != true){
        PyErr_SetString(PyExc_ValueError, "pyobj__read_csv(), arguments require (str, memory obj, dict)");
        return(NULL);
    }
    
    cpp__read_csv(arg_str, &fobj);

    // TODO: infer from first line + samples
    no_columns = fobj.dims[1];
    
    // assume first datatype in list to be the default
    // TODO: infer a datatype default based on pre-sampling
    *column_default = 0;
        
    columns = (uint8_t*) malloc(sizeof(uint8_t) * (no_columns+1));

    // check for a default datatype
    if((dtype_dict_obj = PyDict_GetItemString(dtype_dict, "__default__"))){
        if(PyUnicode_Check(dtype_dict_obj) == true){
            if(!(tmp=(char*)PyUnicode_DATA(dtype_dict_obj))){
                PyErr_SetString(PyExc_ValueError, "__default__ key must be a string.");
                return(NULL);
            }
            _datatype_column_typebyname(tmp, column_default);

            for(j_col=0; j_col<no_columns; j_col++){ columns[j_col] = column_default[0];}

        }else{
            // other types not yet supported
            PyErr_SetString(PyExc_ValueError, "__default__ key must be a string.");
            return(NULL);
        }
    }

    // check for per column settings
    if((dtype_dict_obj = PyDict_GetItemString(dtype_dict, "__columns__"))){
        if(PyDict_Check(dtype_dict_obj) == true){
            // TODO. Parse based on column names
            PyErr_SetString(PyExc_ValueError, "__columns__ dict object not yet supported.");
            return(NULL);
        }
        else if(PyList_Check(dtype_dict_obj) == true){
            // Parse based on list of datatypes
            if(PyList_Size(dtype_dict_obj) != (Py_ssize_t) no_columns){
                PyErr_SetString(
                        PyExc_ValueError,
                        ("__columns__ number of datatypes does not match number of columns. Required:" + std::to_string(no_columns)).c_str()
                );
                return(NULL);
            }
            for(j_col=0; j_col<no_columns; j_col++){
                if(!(dtype_tmp = PyList_GetItem(dtype_dict_obj, (Py_ssize_t) j_col))
                    || (PyUnicode_Check(dtype_tmp) != true)
                    || !(tmp=(char*)PyUnicode_DATA(dtype_tmp))
                ){
                    PyErr_SetString(
                        PyExc_ValueError,
                        ("__columns__ cant retrieve list item_no:" + std::to_string(j_col)).c_str()
                    );

                }
                if(_datatype_column_typebyname(tmp, &columns[j_col]) != true){
                    PyErr_SetString(
                        PyExc_ValueError,
                        ("__columns__ cant find datatype match on column:" + std::to_string(j_col)).c_str()
                    );
                    return(NULL); 
                }
            }
        }
    }

    columns[no_columns] = pow(2, 8) - 1;
    
    dest = (void*) PyCapsule_GetPointer(capsule_mem, "hostmemory_ptr");

    // Parse raw data and put it in location dest
    cpp__parse_csv(&fobj, dest, columns);
    
    PyObject *array_list;
	cpp__new_arraylist(dest, &array_list, fobj.dims[1], fobj.dims[0], columns);
    
	free(fobj.offsets);
    free(fobj.dims);

    return array_list;
};


PyObject* pyobj__write_csv(PyObject* self, PyObject* args){
// TODO: function does not yet contain proper eror checking.
    PyObject *input_str;
    PyObject *input_frame;
    PyObject *pyobj;

    PyObject *list_item_0;
    PyObject *list_item_1;

    Py_buffer *memory_buf;
    void *memory_ptr;

    char *outputfile;
    char *tmp_ptr;

    uint64_t ii;
    uint64_t no_columns;
    uint64_t no_rows;
   
    uint64_t *ptr_data;
    uint64_t *ptr_na;
    uint8_t *datatypes;
    uint8_t *strides;
    uint8_t *functionmap;

    // Verify input
    if(!PyArg_ParseTuple(args, "OO", &input_str, &input_frame)
        || PyUnicode_Check(input_str) != true
        || !(outputfile=(char*)PyUnicode_DATA(input_str))
        || PyDict_Check(input_frame) != true
    ){
        PyErr_SetString(PyExc_ValueError, "pyobj__write_csv() requires (str, tframe dict) as input.");
        return(NULL);
    }

    // Extract no_columns
    if(!(pyobj = PyDict_GetItemString(input_frame, "no_columns"))
        || PyLong_Check(pyobj) != true
        || sizeof(PyLong_AsLong(pyobj)) != sizeof(uint64_t)
    ){
        PyErr_SetString(
            PyExc_ValueError,
            "pyobj__write_csv(). Error parsing \"no_columns\" value"
        );
        return(NULL);
    }
    no_columns = (uint64_t) PyLong_AsLong(pyobj);

    // Extract no_rows
    if(!(pyobj = PyDict_GetItemString(input_frame, "no_rows"))
        || PyLong_Check(pyobj) != true
        || sizeof(PyLong_AsLong(pyobj)) != sizeof(uint64_t)
    ){
        PyErr_SetString(
            PyExc_ValueError,
            "pyobj__write_csv(). Error parsing \"no_rows\" value"
        );
        return(NULL);
    }
    no_rows = (uint64_t) PyLong_AsLong(pyobj);

    // Extract datatypes
    if(!(pyobj = PyDict_GetItemString(input_frame, "datatypes"))
        || PyList_Check(pyobj) != true
        || (uint64_t) PyList_Size(pyobj) != no_columns
    ){
        PyErr_SetString(
            PyExc_ValueError,
            "pyobj__write_csv(). Error parsing \"datatypes\" value"
        );
        return(NULL);
    }

        // Extract column (data-) types
    datatypes = (uint8_t*) malloc(sizeof(uint8_t) * no_columns);
    strides = (uint8_t*) malloc(sizeof(uint8_t) * no_columns);
    functionmap = (uint8_t*) malloc(sizeof(uint8_t) * no_columns);

    for(ii = 0; ii < no_columns; ii++){
        if(!(list_item_0 = PyList_GetItem(pyobj, (Py_ssize_t) ii))
            || (PyUnicode_Check(list_item_0) != true)
            || PyUnicode_GET_LENGTH(list_item_0) > 4
            || !(tmp_ptr=(char*)PyUnicode_DATA(list_item_0))
            || _datatype_column_typebyname(tmp_ptr, &datatypes[ii]) != true
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__write_csv(). Error parsing datatype, list_no: " + std::to_string(ii)).c_str()
            );
            free(datatypes);
            free(strides);
            free(functionmap);
            return(NULL);
        }
        functionmap[ii] = _datatype_column_nobyid(datatypes[ii]);
        strides[ii] = TF_TYPES[functionmap[ii]].size;
    }

    // Extract memory pointers to both NaN and Data
    if(!(pyobj = PyDict_GetItemString(input_frame, "memory_ptr"))
        || PyList_Check(pyobj) != true
        || (uint64_t) PyList_Size(pyobj) != no_columns
    ){
        PyErr_SetString(
            PyExc_ValueError,
            "pyobj__write_csv(). Error parsing \"memory_ptr\" value"
        );
        free(datatypes);
        free(strides);
        free(functionmap);
        return(NULL);
    }

    ptr_data = (uint64_t*) malloc(sizeof(uint64_t) * no_columns);
    ptr_na = (uint64_t*) malloc(sizeof(uint64_t) * no_columns);
    for(ii = 0; ii < no_columns; ii++){
        // memory_ptr is a list of lists (list_item_0 = first level, list_item_1 = second level)
        if(!(list_item_0 = PyList_GetItem(pyobj, (Py_ssize_t) ii))
            || PyList_Check(list_item_0) != true
            || (uint64_t) PyList_Size(list_item_0) != 2     // Note. Expecting format: [NaN ptr, Data ptr]
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__write_csv(). Error parsing \"memory_ptr\", list_no: " + std::to_string(ii)).c_str()
            );
            free(ptr_data);
            free(ptr_na);
            free(datatypes);
            free(strides);
            free(functionmap);
            return(NULL);
        }

        // Retrieve memory point object to NaN
        if(!(list_item_1 = PyList_GetItem(list_item_0, 0))
            || PyMemoryView_Check(list_item_1) != true
            || !(memory_buf = PyMemoryView_GET_BUFFER(list_item_1))
            || !(memory_ptr = (void*) memory_buf->buf)
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__write_csv(). Error parsing \"memory_ptr: NaN\", list_no: " + std::to_string(ii)).c_str()
            );
            free(ptr_data);
            free(ptr_na);
            free(datatypes);
            free(strides);
            free(functionmap);
            return(NULL);
        }
        ptr_na[ii] = (uint64_t) memory_ptr;

        // Retrieve memory point object to Data
        if(!(list_item_1 = PyList_GetItem(list_item_0, 1))
            || PyMemoryView_Check(list_item_1) != true
            || !(memory_buf = PyMemoryView_GET_BUFFER(list_item_1))
            || !(memory_ptr = (void*) memory_buf->buf)
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__write_csv(). Error parsing \"memory_ptr: Data\", list_no: " + std::to_string(ii)).c_str()
            );
            free(ptr_data);
            free(ptr_na);
            free(datatypes);
            free(strides);
            free(functionmap);
            return(NULL);
        }
        ptr_data[ii] = (uint64_t) memory_ptr;
    }

    // Create the frame object for passing it to our writer
    tf__frame *frame = (tf__frame*) malloc(sizeof(tf__frame));
    if(!frame){ 
        free(ptr_data);
        free(ptr_na);
        free(datatypes);
        free(strides);
        free(functionmap);
        return PyErr_NoMemory();
    };

    frame->no_columns = no_columns;
    frame->no_rows = no_rows;
    frame->ptr_data = ptr_data;
    frame->ptr_na = ptr_na;
    frame->datatypes = datatypes;
    frame->strides = strides;
    frame->functionmap = functionmap;

    // Write to csv
    try{ cpp__write_csvfile(outputfile, frame);}
    catch(int error_code){
        PyErr_SetString(
            PyExc_ValueError,
            ("pyobj__write_csv()::cpp_write_csvfile(),error=" + to_string(error_code)).c_str()
        );
        free(ptr_data);
        free(ptr_na);
        free(datatypes);
        free(strides);
        free(functionmap);
        free(frame);
        return(NULL);
    }

    free(ptr_data);
    free(ptr_na);
    free(datatypes);
    free(strides);
    free(functionmap);
    free(frame);

    // Return success
    return Py_BuildValue("i", 0);
}


PyObject* pyobj__generate_arrays(PyObject* self, PyObject* args){
// TODO: function does not yet contain proper eror checking.
    PyObject *input_memory_ptr;
    PyObject *input_rows;
    PyObject *input_list;
    PyObject *list_item;
    PyObject *dict_item;
    PyObject *array_list;

    uint32_t ii;
    uint32_t no_columns;
    uint8_t datatype;
    uint8_t *columns;
    
    uint64_t data_offset;
    uint8_t column_size;
    uint64_t offset_round;;

    void *dest;
    char *char_ptr;

    uint64_t no_rows;
    uint64_t min_value;
    uint64_t max_value;

    // Parse all input arguments into PyObjects
    if(!PyArg_ParseTuple(args, "OOO", &input_memory_ptr, &input_rows, &input_list)
        || PyCapsule_CheckExact(input_memory_ptr) != true
        || PyLong_Check(input_rows) != true
        || PyList_Check(input_list) != true
        || sizeof(PyLong_AsLong( input_rows)) != sizeof(uint64_t)
    ){
        PyErr_SetString(PyExc_ValueError, "pyobj__generate_arrays() requires (pointer, long, list) as an input.");
        return(NULL);
    }

    no_rows = (uint64_t) PyLong_AsLong(input_rows);

    if((uint64_t) PyList_Size(input_list) > UINT32_MAX){
        PyErr_SetString(
            PyExc_ValueError,
            ("pyobj__generate_arrays(). Input list exceeded max number of arrays. Max:" + std::to_string(UINT32_MAX)).c_str()
        );
        return(NULL);
    }
    if((no_columns = (uint32_t) PyList_Size(input_list)) == 0){
        Py_INCREF(Py_None);
        return Py_None;
    }

    // Allocate a data structure to hold column information
    tf_column_meta *columns_meta = (tf_column_meta*) malloc(no_columns * sizeof(tf_column_meta));
    if(!columns_meta){ return PyErr_NoMemory();};
    
    columns = (uint8_t*) malloc(sizeof(uint8_t) * no_columns);

    data_offset = 0;
    offset_round = 0;
    for(ii = 0; ii < no_columns; ii++){
        // List items should be dicts
        if(!(list_item = PyList_GetItem(input_list, ii))
            || PyDict_Check(list_item) != true
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__generate_arrays(). Error parsing item from list, list_no: " + std::to_string(ii)).c_str()
            );
            free(columns);
            free(columns_meta);
            return(NULL);
        }
   
        // Retrieve datatype 
        if(!(dict_item = PyDict_GetItemString(list_item, "datatype"))
            || PyUnicode_Check(dict_item) != true
            || PyUnicode_GET_LENGTH(dict_item) > 4
            || !(char_ptr=(char*)PyUnicode_DATA(dict_item))
            || _datatype_column_typebyname(char_ptr, &datatype) != true
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__generate_arrays(). Error parsing datatype, list_no: " + std::to_string(ii)).c_str()
            );
            free(columns);
            free(columns_meta);
            return(NULL);
        }

        // Retrieve min_value
        if(!(dict_item = PyDict_GetItemString(list_item, "min_value"))
            || PyLong_Check(dict_item) != true
            || sizeof(PyLong_AsLong(dict_item)) != sizeof(uint64_t)
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__generate_arrays(). Error parsing min_value, list_no: " + std::to_string(ii)).c_str()
            );
            free(columns);
            free(columns_meta);
            return(NULL);
        }
        min_value = (uint64_t) PyLong_AsLong(dict_item);


        // Retrieve max_value
        if(!(dict_item = PyDict_GetItemString(list_item, "max_value"))
            || PyLong_Check(dict_item) != true
            || sizeof(PyLong_AsLong(dict_item)) != sizeof(uint64_t)
        ){
            PyErr_SetString(
                PyExc_ValueError,
                ("pyobj__generate_arrays(). Error parsing max_value, list_no: " + std::to_string(ii)).c_str()
            );
            free(columns);
            free(columns_meta);
            return(NULL);
        }
        max_value = (uint64_t) PyLong_AsLong(dict_item);

        // Write to column_meta struct
        columns_meta[ii].datatype = datatype;
        columns_meta[ii].no_items = no_rows;
        columns_meta[ii].min_value = min_value;
        columns_meta[ii].max_value = max_value;
        columns_meta[ii].offset = data_offset;

        columns[ii] = datatype;

        // Calculate offsets for next round
        column_size = TF_TYPES[_datatype_column_nobyid(datatype)].size;
        offset_round = (8 - ((column_size * no_rows) % 8));

        // Begin of NA
        data_offset = data_offset + (no_rows * column_size) + offset_round;
        columns_meta[ii].offset_na = data_offset;
        // End of NA
        data_offset = data_offset + ((no_rows / 32) * 4) + (8 - (((no_rows / 32) * 4) % 8));
    }
    
    dest = (void*) PyCapsule_GetPointer(input_memory_ptr, "hostmemory_ptr");

    cu__generate_arrays(columns_meta, dest, (uint64_t) no_columns, no_rows);    //, )stride_dest, no_columns, no_items);
    free(columns_meta);

	cpp__new_arraylist(dest, &array_list, no_columns, no_rows, columns);
    free(columns);

    return array_list;
}



PyMODINIT_FUNC
PyInit_libio(void)
{
    import_array();
    return PyModule_Create(&cModPyDem_libio);
}


void cpp__new_arraylist(void *data, PyObject **list_ptr, uint64_t no_columns, uint64_t no_rows, uint8_t *columns){
// TODO: function does not yet contain proper eror checking.
    PyObject *array_list;
    PyObject *tmp_tuple;
    PyObject *op;
    PyArray_Descr *descr;
    PyObject *result_col;
    PyObject *result_na;

    uint64_t data_offset;
    uint8_t column_size;
    uint64_t offset_round;;
    char column_name[32];

    npy_intp dimension[1];
    npy_intp dimension_na[1];

    dimension[0] = (int) no_rows;
    dimension_na[0] = (int) (no_rows / 32 * 1 + 1);

    npy_intp strides[1];

    array_list = PyList_New((Py_ssize_t) no_columns);

    data_offset = 0;
    offset_round = 0;
    for(uint64_t j_col = 0; j_col < no_columns; j_col++){
        //COLUMNS 
        sprintf(column_name, "COL_%lu", j_col);

        op = Py_BuildValue("[(s,s)]", column_name, TF_TYPES[_datatype_column_nobyid(columns[j_col])].name);
        PyArray_DescrConverter(op, &descr);
        Py_DECREF(op);

        column_size = TF_TYPES[_datatype_column_nobyid(columns[j_col])].size;
        strides[0] = column_size;
        result_col = PyArray_NewFromDescr(&PyArray_Type, descr, 1, dimension, strides, ((char*) data + data_offset), 0, NULL);

        offset_round = (8 - ((column_size * no_rows) % 8));

        //Begin of NA
        data_offset = data_offset + (no_rows * column_size) + offset_round;

        //NA COLUMNS
        op = Py_BuildValue("[(s,s)]", "", "<i4");
        PyArray_DescrConverter(op, &descr);
        Py_DECREF(op);
        result_na = PyArray_NewFromDescr(&PyArray_Type, descr, 1, dimension_na, NULL, ((char*) data + data_offset), 0, NULL);

        //PUT INTO LIST
        tmp_tuple = PyTuple_New(2);
        PyTuple_SetItem(tmp_tuple, 0, (PyObject *) result_col);
        PyTuple_SetItem(tmp_tuple, 1, (PyObject *) result_na);
        PyList_SetItem(array_list, j_col, (PyObject *) tmp_tuple);

        data_offset = data_offset + ((no_rows / 32) * 4) + (8 - (((no_rows / 32) * 4) % 8));
    }
    *list_ptr = array_list;
}


