#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

//#include <mcpl.h>
/*int not_doublematrix(PyArrayObject *mat) {*/
/*   if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2) {*/
/*      PyErr_SetString(PyExc_ValueError,*/
/*         "In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");*/
/*      return 1; }*/
/*   return 0;*/
/*}*/

static PyObject *numpy2mcpl_dump(PyObject *self, PyObject *args){
  const char *filename;
  PyArrayObject *particle_bank;

  int sts,n,m;
  int dims[2];

  if (!PyArg_ParseTuple(args, "sO!", &filename, &PyArray_Type, &particle_bank))
    return NULL;
  //if (!PyArg_ParseTuple(args, "s", &filename))
  /* Check that object input is 'double' type and a matrix
     Not needed if python wrapper function checks before call to this routine.
     Also, ideally should allow float*/
  //if (not_doublematrix(particle_bank)) return NULL;
  /* Get the dimensions of the input */
  //n=dims[0]=particle_bank->dimensions[0];
  //m=dims[1]=particle_bank->dimensions[1];
  n=12;
  m=45;
  printf("%s %d %d\n",filename, n,m);
  sts = 0;//n*m;
  return PyLong_FromLong(sts);
}



static PyMethodDef mymethods[] = {
    { "numpy2mcpl_dump", numpy2mcpl_dump,
      METH_VARARGS,
      "dump particle data in the form of a numpy array to mcpl"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static char numpy2mcpl_doc[] = "generate an mcpl-file from a numpy array";


static struct PyModuleDef numpy2mcpl = {
  PyModuleDef_HEAD_INIT,
  "nump2mcpl",
  numpy2mcpl_doc,
  -1,
  mymethods
};

static PyObject *n2m_Error;

PyMODINIT_FUNC
PyInit_numpy2mcpl(void)
{
  PyObject *m;
  m=PyModule_Create(&numpy2mcpl);

  if (m == NULL)
        return NULL;

  n2m_Error = PyErr_NewException("numpy2mcpl.error", NULL, NULL);
  Py_XINCREF(n2m_Error);
  if (PyModule_AddObject(m, "error", n2m_Error) < 0) {
    Py_XDECREF(n2m_Error);
    Py_CLEAR(n2m_Error);
    Py_DECREF(m);
    return NULL;
  }
  return m;
}

