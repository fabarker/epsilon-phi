#include "Python.h"
#include "numpy/arrayobject.h"
#include <iostream>
#include <vector>
#include <cstdio>
#include <set>


template<class REAL>
inline
std::vector<int> gkse(REAL* x,
                      REAL* limit,
                      long n) {

    long m = 3; // Fixed number of columns
    std::vector<int> results(n, -1);

    std::set<REAL> unique_elements(limit, limit + n);
    // Iterating over each unique element
    for (const REAL& unique_value : unique_elements) {
        std::cout << "Unique value: " << unique_value << std::endl;

        // For each unique value, iterate over the limit array
        for (long i = 0; i < n; ++i) {
            // If the current value in limit matches the unique value
            if (limit[i] == unique_value) {
                // Print the corresponding row values from the x array
                std::cout << "Matching row in x at index " << i << ": ";
                for (long j = 0; j < m; ++j) { // Loop through columns
                    // Correctly calculate the index for a 1D array representing a 2D structure
                    std::cout << x[i * m + j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    return results; // Return the vector of results
}

static PyObject *cc_kse(PyObject *self, PyObject *args);
struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyMethodDef kse_methods[] = {
  {"kse", cc_kse, METH_VARARGS,"kse(arr, lim, op)\nreturns element to first index ind for that\n\n arr[i] OP(op) limit\n\n is True.\nOP(op) represents one of the comparison operators defined as follows:\n\n   OP(-2) -> < \n   OP(-1) -> <= \n   OP(0) ->  == \n   OP(1) -> >= \n   OP(2) -> >\n"},
  {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3

static int kse_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int kse_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "kse",
        NULL,
        sizeof(struct module_state),
        kse_methods,
        NULL,
        kse_traverse,
        kse_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_kse(void)
#else
#define INITERROR return

PyMODINIT_FUNC
initkse(void)
#endif

{
   import_array();
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("kse", kse_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);
    char exception_text[16] = "kse.Error";
    st->error = PyErr_NewException(exception_text, NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

static PyObject *cc_kse(PyObject *dummy, PyObject *args) {

  PyArrayObject *array_;
  PyArrayObject *group;
  if (!PyArg_ParseTuple(args, "O!O!:kse",
                        &PyArray_Type, &array_,
                        &PyArray_Type, &group))
                        return NULL;

  int numDimsLimit = PyArray_NDIM((PyArrayObject*)array_);

  int stride_x = PyArray_STRIDE(array_, 0);
  int pytype   = PyArray_TYPE(array_);
  int n = PyArray_SIZE(group);

  std::vector<int> ret;

  switch(pytype) {
  case NPY_DOUBLE:
    ret = gkse(reinterpret_cast<double*>(PyArray_DATA(group)),
               reinterpret_cast<double*>(PyArray_DATA(array_)),
               static_cast<long>(PyArray_DIMS(group)[0]));
    break;
  case NPY_FLOAT:
    ret = gkse(reinterpret_cast<float*>(PyArray_DATA(group)),
               reinterpret_cast<float*>(PyArray_DATA(array_)),
               static_cast<long>(PyArray_DIMS(group)[0]));
    break;
  case NPY_INT64:
    ret = gkse(reinterpret_cast<long*>(PyArray_DATA(group)),
               reinterpret_cast<long*>(PyArray_DATA(array_)),
               static_cast<long>(PyArray_DIMS(group)[0]));
    break;
  case NPY_INT32:
    ret = gkse(reinterpret_cast<int*>(PyArray_DATA(group)),
               reinterpret_cast<int*>(PyArray_DATA(array_)),
               static_cast<long>(PyArray_DIMS(group)[0]));
    break;
  default:
  PyErr_SetString(PyExc_ValueError,
                 "cc_kse::Input data type must be one of float64, float32, int64, int32, or bool.");
       return NULL;
  }

  return NULL;
}

