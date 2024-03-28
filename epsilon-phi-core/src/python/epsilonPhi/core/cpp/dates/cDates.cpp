#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <datetime.h>
#include <stdexcept> // For std::invalid_argument and std::out_of_range
#include <iostream>
#include <string>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <vector> // Include this to use std::vector

static const int _days_in_month[] = {
    0, /* unused; this vector uses 1-based indexing */
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

static const int _days_before_month[] = {
    0, /* unused; this vector uses 1-based indexing */
    0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334
};

/* year -> 1 if leap year, else 0. */
static int is_leap(int year) {
    const unsigned int ayear = (unsigned int)year;
    return ayear % 4 == 0 && (ayear % 100 != 0 || ayear % 400 == 0);
}

static int
days_in_month(int year, int month)
{
    assert(month >= 1);
    assert(month <= 12);
    if (month == 2 && is_leap(year))
        return 29;
    else
        return _days_in_month[month];
}

/* year, month -> number of days in year preceding first day of month */
static int days_before_month(int year, int month) {
    int days;

    assert(month >= 1);
    assert(month <= 12);
    days = _days_before_month[month];
    if (month > 2 && is_leap(year))
        ++days;
    return days;
}

/* year -> number of days before January 1st of year.  Remember that we
 * start with year 1, so days_before_year(1) == 0.
 */
static int days_before_year(int year) {
    int y = year - 1;
    /* This is incorrect if year <= 0; we really want the floor
     * here.  But so long as MINYEAR is 1, the smallest year this
     * can see is 1.
     */
    assert (year >= 1);
    return y*365 + y/4 - y/100 + y/400;
}

/* Number of days in 4, 100, and 400 year cycles.  That these have
 * the correct values is asserted in the module init function.
 */
#define DI4Y    1461    /* days_before_year(5); days in 4 years */
#define DI100Y  36524   /* days_before_year(101); days in 100 years */
#define DI400Y  146097  /* days_before_year(401); days in 400 years  */

static void ord_to_ymd(int ordinal, int *year, int *month, int *day) {
    int n, n1, n4, n100, n400, leapyear, preceding;

    assert(ordinal >= 1);
    --ordinal;
    n400 = ordinal / DI400Y;
    n = ordinal % DI400Y;
    *year = n400 * 400 + 1;

    /* Now n is the (non-negative) offset, in days, from January 1 of
     * year, to the desired date.  Now compute how many 100-year cycles
     * precede n.
     * Note that it's possible for n100 to equal 4!  In that case 4 full
     * 100-year cycles precede the desired day, which implies the
     * desired day is December 31 at the end of a 400-year cycle.
     */
    n100 = n / DI100Y;
    n = n % DI100Y;

    /* Now compute how many 4-year cycles precede it. */
    n4 = n / DI4Y;
    n = n % DI4Y;

    /* And now how many single years.  Again n1 can be 4, and again
     * meaning that the desired day is December 31 at the end of the
     * 4-year cycle.
     */
    n1 = n / 365;
    n = n % 365;

    *year += n100 * 100 + n4 * 4 + n1;
    if (n1 == 4 || n100 == 4) {
        assert(n == 0);
        *year -= 1;
        *month = 12;
        *day = 31;
        return;
    }

    /* Now the year is correct, and n is the offset from January 1.  We
     * find the month via an estimate that's either exact or one too
     * large.
     */
    leapyear = n1 == 3 && (n4 != 24 || n100 == 3);
    assert(leapyear == is_leap(*year));
    *month = (n + 50) >> 5;
    preceding = (_days_before_month[*month] + (*month > 2 && leapyear));
    if (preceding > n) {
        /* estimate is too large */
        *month -= 1;
        preceding -= days_in_month(*year, *month);
    }
    n -= preceding;
    assert(0 <= n);
    assert(n < days_in_month(*year, *month));

    *day = n + 1;
}

static int ymd_to_ord(int year, int month, int day) {
    return days_before_year(year) + days_before_month(year, month) + day;
}

static int weekday(int year, int month, int day) {
    return (ymd_to_ord(year, month, day) + 6) % 7;
}



std::vector<int> shift_off_weekend(int Y, int M, int D, int rollover) {

    std::vector<int> newDate(3, -1);
    int wkday = weekday(Y, M, D);
    int ord = ymd_to_ord(Y, M, D);

    if (wkday == 6) { // Saturday
        // Check if adding two days shifts to next month
        if ((D + 2 > days_in_month(Y, M)) && (rollover == -1)) {
            // Shift back to the last Friday of the current month
            D -= 1;
        } else {
            ord += 2;
            ord_to_ymd(ord, &Y, &M, &D);
        }
    } else if (wkday == 0) { // Sunday
        // Check if adding one day shifts to next month
        if ((D + 1 > days_in_month(Y, M)) && (rollover == -1)) {
            // Shift back to the last Friday
            D -= 2;
        } else {
            ord += 1;
            ord_to_ymd(ord, &Y, &M, &D);
        }
    }

    // Here you should include logic to correctly handle month and year rollover
    // For simplicity, it's omitted
    newDate[0] = Y;
    newDate[1] = M;
    newDate[2] = D;
    return newDate;
}

static int lsbusday(int Y, int M){
    int DM = days_in_month(Y, M); // Get the last day of the month
    std::vector<int> weekday = shift_off_weekend(Y, M, DM, -1); // Correctly append the missing semicolon here
    return weekday[2]; // Return the day component, which is adjusted to be a business day
}

int extractInteger(const std::string& input) {
    try {
        size_t pos;
        int result = std::stoi(input, &pos);
        // If needed, use 'pos' to perform further checks, e.g., to verify that the entire string was converted
        return result;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid input: no conversion could be performed.\n";
    } catch (const std::out_of_range& e) {
        std::cerr << "Invalid input: the converted value is out of range for an int.\n";
    }

    return 0; // Return a default value or handle error as appropriate
}

char extractTimeUnit(const std::string& inputString) {
    // Find the first non-digit character in the input string
    std::size_t pos = inputString.find_first_not_of("0123456789");
    if (pos != std::string::npos) {
        // If found, return the character at this position
        return inputString[pos];
    } else {
        // If no non-digit character is found, return a default indicator (e.g., '\0' to indicate an error)
        return '\0';
    }
}


std::vector<int> addtodate(int Y, int M, int D, int units, char unitType) {
    // This example simply increments the year based on units,
    // but you should adjust the logic to handle different unit types correctly.
    int resultYear = Y, resultMonth = M, resultDay = D;
    int dir = unitType == 'w' || unitType == 'd' ? 1 : -1;
    int ord = 0;
    int lbd = 0;
    int lbd_MY = 0;

    switch (unitType) {
        case 'y': // Year
            resultYear += units;
            lbd = lsbusday(resultYear, M);
            if (D > lbd) {
                resultDay = lbd;
            }
            break;
        case 'm':
            lbd = lsbusday(Y, M);
            resultMonth += units;
            while (resultMonth > 12) {
                resultMonth -= 12;
                resultYear += 1;
            }

            lbd_MY = lsbusday(resultYear, resultMonth);
            if (D == lbd || D > lbd_MY) {
                resultDay = lbd_MY;
            }
            break;
        case 'w':
            ord = ymd_to_ord(Y, M, D) + 7 * units; // Convert weeks to days and calculate the new ordinal date
            ord_to_ymd(ord, &resultYear, &resultMonth, &resultDay); // Convert back to Y, M, D
            break;
        case 'd': // Day
            ord = ymd_to_ord(Y, M, D) + units; // Calculate the new ordinal date
            ord_to_ymd(ord, &resultYear, &resultMonth, &resultDay); // Convert back to Y, M, D
            break;
        default:
            // Perhaps handle unsupported unitType or error
            break;
    }
    std::vector<int> adjustedDate = shift_off_weekend(resultYear, resultMonth, resultDay, dir);
    return adjustedDate; // Return the adjusted date
}

static PyObject* get_expiry_date(PyObject* self, PyObject* args)
{
    PyDateTime_DateTime *pydate;
    const char* inputString;

    if(!PyArg_ParseTuple(args, "Os", &pydate, &inputString)){
        return Py_BuildValue("s", "Error");
    }

    char type = extractTimeUnit(inputString);
    int units = extractInteger(inputString);
    int Y = PyDateTime_GET_YEAR(pydate);
    int M = PyDateTime_GET_MONTH(pydate);
    int D = PyDateTime_GET_DAY(pydate);

    std::vector<int> resultDate = addtodate(Y, M, D, units, type);
    PyObject* pyResultDate = PyDate_FromDate(
            resultDate[0],
            resultDate[1],
            resultDate[2]);

    return pyResultDate;
}

static PyObject* from_ordinals(PyObject* self, PyObject* args) {

    PyArrayObject *ordinals;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &ordinals))
        return NULL;

    npy_intp n = PyArray_SIZE(ordinals);

    PyArrayObject* datetimes = (PyArrayObject*) PyArray_SimpleNew(1, &n, NPY_OBJECT);
    if (!datetimes) return PyErr_NoMemory();
    PyObject** pydates = (PyObject**) PyArray_DATA(datetimes);

    for (npy_intp i = 0; i < n; i++) {
        long ordinal = *(long*)PyArray_GETPTR1(ordinals, i);

        int Y = 0;
        int M = 0;
        int D = 0;
        ord_to_ymd(ordinal, &Y, &M, &D);
        PyObject* pydate = PyDate_FromDate(Y, M, D);

        if (!pydate) {
            Py_DECREF(datetimes);
            return NULL;
        }

        PyArray_SETITEM(datetimes, (char*)(pydates + i), pydate);
        Py_DECREF(pydate); // Reduce reference count after adding to array
    }

    return (PyObject*) datetimes;
}

static PyObject* to_ordinals(PyObject* self, PyObject* args) {

    PyArrayObject *dates;
    if (!PyArg_ParseTuple(args, "O!",
                        &PyArray_Type, &dates))
                        return NULL;

    npy_intp n = PyArray_SIZE(dates);
    npy_intp i;

    // Create a new 1D NumPy array of type int32 to store the ordinals
    PyObject* py_ordinals = PyArray_SimpleNew(1, &n, NPY_INT32);
    if (!py_ordinals) return NULL; // Check if array creation was successful

    for (i = 0; i < n; i++) {
        PyObject* pydate = *(PyObject**)PyArray_GETPTR1(dates, i);
        if (!PyDateTime_Check(pydate)) {
            Py_DECREF(py_ordinals);
            PyErr_SetString(PyExc_TypeError, "Array must contain datetime objects.");
            return NULL;
        }
        int Y = PyDateTime_GET_YEAR(pydate);
        int M = PyDateTime_GET_MONTH(pydate);
        int D = PyDateTime_GET_DAY(pydate);
        int ordinal = ymd_to_ord(Y, M, D);

        int* array_ptr = (int*)PyArray_GETPTR1((PyArrayObject*)py_ordinals, i);
        *array_ptr = ordinal;
    }

    // Return the NumPy array with ordinals
    return py_ordinals;
}


static PyObject* shiftdates(PyObject* self, PyObject* args) {

    PyArrayObject *datevec;
    PyObject * matstr;

    if (!PyArg_ParseTuple(args, "O!O!",
                        &PyArray_Type, &datevec,
                        &PyList_Type, &matstr))
                        return NULL;

    npy_intp n = PyArray_SIZE(datevec);
    npy_intp i;

    PyArrayObject* expiries = (PyArrayObject*) PyArray_SimpleNew(1, &n, NPY_OBJECT);
    if (!expiries) return PyErr_NoMemory();
    PyObject** expiriesData = (PyObject**) PyArray_DATA(expiries);

    for (i = 0; i < n; i++) {

        PyObject* datetime_obj = *(PyObject**)PyArray_GETPTR1(datevec, i);
        PyObject* strObj = PyList_GetItem(matstr, i); // Borrowed reference

        // Ensure correct reference counting
        Py_INCREF(datetime_obj);
        Py_INCREF(strObj);

        PyObject* argsTuple = PyTuple_New(2);
        PyTuple_SetItem(argsTuple, 0, datetime_obj);
        PyTuple_SetItem(argsTuple, 1, strObj);

        PyObject* expiry = get_expiry_date(self, argsTuple);

        // Check if get_expiry_date succeeded
        if (!expiry) {
            Py_DECREF(expiries); // Clean up on failure
            return NULL;
        }

        // Insert expiry into the NumPy array and manage reference count
        PyArray_SETITEM(expiries, (char*)(expiriesData + i), expiry);
        Py_DECREF(expiry); // Reduce reference count after adding to array
    }

    return (PyObject*)expiries;;
}

static PyMethodDef NbourMethods[] = {
     {"get_expiry_date", get_expiry_date, METH_VARARGS, "Do date things."},
     {"shiftdates", shiftdates, METH_VARARGS, "Shift dates based on input strings."},
     {"to_ordinals", to_ordinals, METH_VARARGS, "Single date to ordinals"},
     {"from_ordinals", from_ordinals, METH_VARARGS, "Ordinal to dates"},
     {NULL, NULL, 0, NULL}  // Sentinel value to indicate the end of the array
};

static struct PyModuleDef cDates =
{
  PyModuleDef_HEAD_INIT,
  "cDates", /* name of module */
  "",     /* module documentation */
  -1,     /* size of per-interpreter state of the module or -1 if global */
  NbourMethods
};

PyMODINIT_FUNC PyInit_cDates(void)
{

  import_array(); // For void return type. If failure, it directly triggers a fatal error.
  import_array1(NULL);
  PyDateTime_IMPORT;

  // Check if the import was successful by verifying the API is available.
  if (PyDateTimeAPI == NULL) {
        PyErr_SetString(PyExc_ImportError, "Failed to import the DateTime C API.");
        return NULL;
  }

  if (!PyDateTimeAPI) { PyDateTime_IMPORT;}
  return PyModule_Create(&cDates);
}