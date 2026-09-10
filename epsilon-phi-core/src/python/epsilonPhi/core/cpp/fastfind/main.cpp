#include "Python.h"
#include "numpy/arrayobject.h"
#include <iostream>
#include <vector>
#include <cstdio>

template<class REAL>
inline
std::vector<int> find_1st_templ(REAL* x, REAL* limit,
                                int stride_x, int stride_y,
                                long x_len, long y_len,
                                long width) {

    std::vector<int> results(y_len, -1); // Initialize results vector with -1s

    // Loop through each row in the limit array
    for (long jj = 0; jj < y_len; ++jj) {
        long low = 0;
        long high = x_len - 1;
        while (low <= high) {
            long mid = low + (high - low) / 2;
            bool match = true; // Assume a match until proven otherwise

            // Check each column in the current row for a match
            for (long col = 0; col < width; ++col) {
                REAL curr_x = x[(mid * stride_x) + col];
                REAL curr_y = limit[(jj * stride_y) + col];

                // If any column does not match, this row can't be the one we're looking for
                if (curr_x != curr_y) {
                    match = false;
                    break; // No need to check further columns
                }
            }

            // Perform the binary search logic based on the comparison result
            if (match) {
                results[jj] = mid; // Found a matching row, store the location
                break; // Exit the while loop, move to the next row in 'limit'
            } else {
                // Since we can't use lexicographical_compare directly due to the array and stride,
                // decide to move left or right in the binary search based on manual comparison
                REAL first_x = x[mid * stride_x];
                REAL first_limit = limit[jj * stride_y];
                if (first_x < first_limit) {
                    low = mid + 1; // Search in the right half
                } else {
                    high = mid - 1; // Search in the left half
                }
            }
        }
    }
    return results; // Return the vector of results
}

static PyObject *cc_find_1st(PyObject *self, PyObject *args);


struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyMethodDef find_1st_methods[] = {
  {"find_1st", cc_find_1st, METH_VARARGS,"find_1st(arr, lim, op)\nreturns element to first index ind for that\n\n arr[i] OP(op) limit\n\n is True.\nOP(op) represents one of the comparison operators defined as follows:\n\n   OP(-2) -> < \n   OP(-1) -> <= \n   OP(0) ->  == \n   OP(1) -> >= \n   OP(2) -> >\n"},
  {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3

static int find_1st_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int find_1st_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "find_1st",
        NULL,
        sizeof(struct module_state),
        find_1st_methods,
        NULL,
        find_1st_traverse,
        find_1st_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_find_1st(void)
#else
#define INITERROR return

PyMODINIT_FUNC
initfind_1st(void)
#endif

{
   import_array();
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("find_1st", find_1st_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);
    char exception_text[16] = "find_1st.Error";
    st->error = PyErr_NewException(exception_text, NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

static PyObject *cc_find_1st(PyObject *dummy, PyObject *args) {

  PyArrayObject *limit;
  PyArrayObject *input;

  if (!PyArg_ParseTuple(args, "O!O!:find_1st",
                        &PyArray_Type, &input,
                        &PyArray_Type, &limit))
                        return NULL;

  int numDimsLimit = PyArray_NDIM((PyArrayObject*)limit);
  int numDimsInput = PyArray_NDIM((PyArrayObject*)input);

  if (numDimsLimit != numDimsInput) {
     PyErr_SetString(PyExc_ValueError,
         "cc_find_1st::Limit arrays must have the same dimensions.");
    return NULL;
  }

  if (numDimsLimit > 1) {
        int numColumnsLimit = PyArray_DIM((PyArrayObject*)limit, 1);
        int numColumnsInput = PyArray_DIM((PyArrayObject*)input, 1);

        if (numColumnsLimit != numColumnsInput) {
            PyErr_SetString(PyExc_ValueError,
            "cc_find_1st::Limit arrays must have the same dimensions.");
           return NULL;
        }
  }

  int stride_x = PyArray_STRIDE(input, 0);
  int stride_y = PyArray_STRIDE(limit, 0);
  int pytype   = PyArray_TYPE(input);

  std::vector<int> ret;
  switch(pytype) {
  case NPY_DOUBLE:
    ret = find_1st_templ(reinterpret_cast<double*>(PyArray_DATA(input)),
                         reinterpret_cast<double*>(PyArray_DATA(limit)),
                         stride_x/sizeof(double), stride_y/sizeof(double),
                         static_cast<long>(PyArray_DIMS(input)[0]),
                         static_cast<long>(PyArray_DIMS(limit)[0]),
                         static_cast<long>(PyArray_DIMS(input)[1]));
    break;
  case NPY_FLOAT:
    ret = find_1st_templ(reinterpret_cast<float*>(PyArray_DATA(input)),
                         reinterpret_cast<float*>(PyArray_DATA(limit)),
                         stride_x/sizeof(float), stride_y/sizeof(float),
                         static_cast<long>(PyArray_DIMS(input)[0]),
                         static_cast<long>(PyArray_DIMS(limit)[0]),
                         static_cast<long>(PyArray_DIMS(input)[1]));
    break;
  case NPY_INT64:
    ret = find_1st_templ(reinterpret_cast<long*>(PyArray_DATA(input)),
                         reinterpret_cast<long*>(PyArray_DATA(limit)),
                         stride_x/sizeof(long), stride_y/sizeof(long),
                         static_cast<long>(PyArray_DIMS(input)[0]),
                         static_cast<long>(PyArray_DIMS(limit)[0]),
                         static_cast<long>(PyArray_DIMS(input)[1]));
    break;
  case NPY_INT32:
    ret = find_1st_templ(reinterpret_cast<int*>(PyArray_DATA(input)),
                         reinterpret_cast<int*>(PyArray_DATA(limit)),
                         stride_x/sizeof(int), stride_y/sizeof(int),
                         static_cast<long>(PyArray_DIMS(input)[0]),
                         static_cast<long>(PyArray_DIMS(limit)[0]),
                         static_cast<long>(PyArray_DIMS(input)[1]));
    break;
  case NPY_BOOL:
    ret = find_1st_templ(reinterpret_cast<npy_bool*>(PyArray_DATA(input)),
                         reinterpret_cast<npy_bool*>(PyArray_DATA(limit)),
                         stride_x/sizeof(npy_bool), stride_y/sizeof(npy_bool),
                         static_cast<long>(PyArray_DIMS(input)[0]),
                         static_cast<long>(PyArray_DIMS(limit)[0]),
                         static_cast<long>(PyArray_DIMS(input)[1]));
    break;
  default:
  PyErr_SetString(PyExc_ValueError,
                 "cc_find_1st::Input data type must be one of float64, float32, int64, int32, or bool.");
       return NULL;
  }

  npy_intp dims[1] = {ret.size()}; // npy_intp is the proper integer type for NumPy dimensions
  PyObject* np_ret = PyArray_SimpleNew(1, dims, NPY_INT32); // Create a 1D NumPy array of integers
  if (!np_ret) return NULL; // Check for errors

  int* np_data = static_cast<int*>(PyArray_DATA((PyArrayObject*)np_ret));
  std::copy(ret.begin(), ret.end(), np_data);
  return np_ret;
}

