#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <mcpl.h>

/*int not_doublematrix(PyArrayObject *mat) {*/
/*   if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2) {*/
/*      PyErr_SetString(PyExc_ValueError,*/
/*         "In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");*/
/*      return 1; }*/
/*   return 0;*/
/*}*/



void * failure(PyObject *type, const char *message) {
    PyErr_SetString(type, message);
    return NULL;
}

static PyObject *np2mcpl_save(PyObject *self, PyObject *args){
  char *filename;
  char line[512];

  PyArrayObject *particle_bank;
    
  mcpl_outfile_t outputfile;

  int sts,i,nparticles,m;
  int dims[2];
  int polarised;

  if (!PyArg_ParseTuple(args, "sO!", &filename, &PyArray_Type, &particle_bank))
    return failure(PyExc_RuntimeError, "np2mcpl: Failed to parse parameters.");
  /* We should Check that object input is 'double' type and a matrix
     Also, ideally should allow 'float'*/

  /* Get the dimensions of the input */
  nparticles=dims[0]=particle_bank->dimensions[0];
  m=dims[1]=particle_bank->dimensions[1];
  
  /*create the mcpl output structure*/
  outputfile = mcpl_create_outfile(filename);
  snprintf(line,255,"%s %s","np2mcpl","v0.01");
  mcpl_hdr_set_srcname(outputfile,line);
  
  /*for now always assume double precision*/
  mcpl_enable_doubleprec(outputfile);

  /* check if dims match polaized or not */
  if (m==12) {
    printf("INFO: polarization enabled.\n");
    polarised=1;
    mcpl_enable_polarisation(outputfile);
  } else if (m==9){
    printf("INFO: polarization disabled.\n");
    polarised=0;

  } else {
    printf("ERROR: wrong number of columns in numpy array");
    return failure(PyExc_RuntimeError, "Wrong number of of columns: ({m}. Expected 9 or 12.");
  }


  /* loop over rows in the numpy array and drop everything to an mcpl-file*/
  for (i=0;i<nparticles;i++){
    mcpl_particle_t p;
    p.pdgcode=2112;
    p.position[0]=*( (double *) PyArray_GETPTR2(particle_bank,i,0));
    p.position[1]=*( (double *) PyArray_GETPTR2(particle_bank,i,1));
    p.position[2]=*( (double *) PyArray_GETPTR2(particle_bank,i,2));
    p.direction[0]=*( (double *) PyArray_GETPTR2(particle_bank,i,3));
    p.direction[1]=*( (double *) PyArray_GETPTR2(particle_bank,i,4));
    p.direction[2]=*( (double *) PyArray_GETPTR2(particle_bank,i,5));
    p.time=*( (double *) PyArray_GETPTR2(particle_bank,i,6));
    p.ekin=*( (double *) PyArray_GETPTR2(particle_bank,i,7));
    p.weight=*( (double *) PyArray_GETPTR2(particle_bank,i,8));
    if(polarised){
      p.polarisation[0]=*( (double *) PyArray_GETPTR2(particle_bank,i,9));
      p.polarisation[1]=*( (double *) PyArray_GETPTR2(particle_bank,i,10));
      p.polarisation[2]=*( (double *) PyArray_GETPTR2(particle_bank,i,11));
    }
    /*write the particle*/
    mcpl_add_particle(outputfile,&p);
  }  
  mcpl_closeandgzip_outfile(outputfile);
  sts = m;
  return PyLong_FromLong(sts);
}

static PyMethodDef mymethods[] = {
    { "save", np2mcpl_save,
      METH_VARARGS,
      "Save particle data in the form of a numpy array to mcpl"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static char np2mcpl_doc[] = "Generate an mcpl-file from a numpy array";

static struct PyModuleDef np2mcpl = {
  PyModuleDef_HEAD_INIT,
  "np2mcpl",
  np2mcpl_doc,
  -1,
  mymethods
};

static PyObject *np2mcpl_Error;

PyMODINIT_FUNC
PyInit_np2mcpl(void)
{
  import_array();
  
  PyObject *m;
  m=PyModule_Create(&np2mcpl);

  if (m == NULL)
    return NULL;

  np2mcpl_Error = PyErr_NewException("np2mcpl.error", NULL, NULL);
  Py_XINCREF(np2mcpl_Error);
  if (PyModule_AddObject(m, "error", np2mcpl_Error) < 0) {
    Py_XDECREF(np2mcpl_Error);
    Py_CLEAR(np2mcpl_Error);
    Py_DECREF(m);
    return NULL;
  }
  return m;
}

