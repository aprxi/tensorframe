/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include <Python.h>
#include <numpy/arrayobject.h>

extern "C" {
    PyObject* pyobj__array_sum(PyObject *self, PyObject *args);

  static PyMethodDef table_Methods[] =
  {
       {"array_sum", pyobj__array_sum, METH_VARARGS, "return the sums of a list of arrays"},
       {NULL, NULL, 0, NULL}
  };

  static struct PyModuleDef cModPyDem_libbinops =
  {
      PyModuleDef_HEAD_INIT,
      "libbinops",
      "\
\n===============================================================\n\
libbinops is a c(++) based extension to the TensorFrame python\n\
module with the purpose to accelerate tensor compute operations.\n\
\n===============================================================\n\
      ",
      -1,
      table_Methods
  };
}

void cu__array_sum(void *columns, void *dest, uint8_t stride_dest, uint64_t no_columns, uint64_t no_items);

