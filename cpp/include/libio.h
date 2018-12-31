/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include <Python.h>
#include <numpy/arrayobject.h>

extern "C" {
    PyObject* pyobj__read_csv(PyObject *self, PyObject *args);
    PyObject* pyobj__write_csv(PyObject *self, PyObject *args);
    PyObject* pyobj__generate_arrays(PyObject *self, PyObject *args);
    PyObject* pyobj__gpu_halloc(PyObject *self, PyObject *args);

  static PyMethodDef table_Methods[] =
  {
       {"read_csv", pyobj__read_csv, METH_VARARGS, "read from csv"},
       {"write_csv", pyobj__write_csv, METH_VARARGS, "write to csv"},
       {"generate_arrays", pyobj__generate_arrays, METH_VARARGS, "generate arrays"},
       {"gpu_halloc", pyobj__gpu_halloc, METH_VARARGS, "allocate gpu pinned hostmemory"},
       {NULL, NULL, 0, NULL}
  };

  static struct PyModuleDef cModPyDem_libio =
  {
      PyModuleDef_HEAD_INIT,
      "libio",
      "\
\n===============================================================\n\
libio is a c(++) based extension to the TensorFrame python\n\
module with the purpose to accelerate tensor compute operations.\n\
\n===============================================================\n\
      ",
      -1,
      table_Methods
  };
}

struct file_object {
    uint64_t fsize;
    uint64_t chunks;
    uint64_t *dims;
    uint64_t *offsets;
    char *string;
};

typedef struct {
    uint64_t no_columns;
    uint64_t no_rows;
    uint64_t *ptr_data;
    uint64_t *ptr_na;
    uint8_t *datatypes;
    uint8_t *strides;
    uint8_t *functionmap;
    void *column_meta;
} tf__frame;

void cpp__new_arraylist(void *data, PyObject **list_ptr, uint64_t no_columns, uint64_t no_rows, uint8_t *columns);
void cpp__read_csv(char *arg_str, struct file_object *fobj);
void cpp__parse_csv(struct file_object *fobj, void *dest, uint8_t *columns);

void cpp__write_csvfile(char *outputfile, tf__frame *frame);

void cuda_parse_csv(char *A, unsigned long fsize, unsigned long *offsets, unsigned long chunks, unsigned long *dims, void *dest, uint8_t *column_types);

void cu__generate_arrays(void *columns_meta, void *dest, uint64_t no_columns, uint64_t no_items);

void cu_malloc_carr(char **data_ptr, unsigned long long fsize);
