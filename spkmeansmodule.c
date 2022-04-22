#include <Python.h>
#define PY_SSIZE_T_CLEAN
#include "spkmeans.h"
int global_rows, global_cols;

static PyObject* None(){
    Py_INCREF(Py_None);
    return Py_None;
}

static double** Py_Arr_To_C(PyObject* flat_py_arr, int rows, int cols) {
    double** array;
    int i, j;
    array = Array_2D(rows, cols, 1);
    if (array == NULL)
        return NULL;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            array[i][j] = PyFloat_AsDouble((PyList_GetItem(flat_py_arr, j + i*cols)));
        }
    }
    return array;
}

static PyObject* C_Arr_To_Py(double** c_array, int rows, int cols) {
    int i, j;
    PyObject* py_arr = PyList_New(rows * cols);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            PyList_SetItem(py_arr, j + i * cols, PyFloat_FromDouble(c_array[i][j]));
        }
    }
    return py_arr;
}


static PyObject* kmeans_capi(PyObject *self, PyObject *args) {
    int K, res, rows, cols;
    double** vectors;
    double** clusters;
    PyObject* flat_vectors;
    PyObject* flat_clusters;
    if (!PyArg_ParseTuple(args, "iiiOO", &K, &rows, &cols, &flat_vectors, &flat_clusters))
        return None();

    vectors = Py_Arr_To_C(flat_vectors, rows, cols);
    clusters = Py_Arr_To_C(flat_clusters, K, cols);
    if (vectors == NULL || clusters == NULL)
        return None();

    res = fit_kmeans(vectors, clusters, K, rows, cols);
    if (res == 1) {
        free_Mem(vectors, rows);
        free_Mem(clusters, K);
        return None();
    }
    PyObject *py_clusters = C_Arr_To_Py(clusters, K, cols);
    free_Mem(vectors, rows);
    free_Mem(clusters, K);
    return py_clusters;
}


static double** Prologue(PyObject *self, PyObject *args) {
    PyObject *flat_vectors;
    if (!PyArg_ParseTuple(args, "iiO", &global_rows, &global_cols, &flat_vectors))
        return None();
    return Py_Arr_To_C(flat_vectors, global_rows, global_cols);
}

static PyObject* Epilogue(double** vectors, int rows, int cols) {
    if (vectors == NULL){
        free_Mem(vectors, rows);
        return None();
    }
    PyObject *py_matrix = C_Arr_To_Py(vectors, rows, cols);
    free_Mem(vectors, rows);
    return py_matrix;
}

static PyObject* spk_capi(PyObject *self, PyObject *args) {
    int k;
    double** vectors = Prologue(self, args);
    if (vectors == NULL)
        return None();
    vectors = spk(vectors, global_rows, global_cols, &k);
    PyObject * python_val = Epilogue(vectors, global_rows, k);
    return Py_BuildValue("iO", k, python_val);
}

static PyObject* wam_capi(PyObject *self, PyObject *args) {
    double** vectors = Prologue(self, args);
    if (vectors == NULL)
        return None();
    vectors = wam(vectors, global_rows, global_cols);
    return Epilogue(vectors, global_rows, global_rows);
}

static PyObject* ddg_capi(PyObject *self, PyObject *args) {
    double** vectors = Prologue(self, args);
    double** W;
    if (vectors == NULL)
        return None();
    W = wam(vectors, global_rows, global_cols);
    vectors = ddg(W, global_rows);
    free_Mem(W, global_rows);
    return Epilogue(vectors, global_rows, global_rows);
}

static PyObject* lnorm_capi(PyObject *self, PyObject *args) {
    double** vectors = Prologue(self, args);
    if (vectors == NULL)
        return None();
    vectors = L_norm(vectors, global_rows, global_cols);
    return Epilogue(vectors, global_rows, global_rows);
}

static PyObject* jacobi_capi(PyObject *self, PyObject *args) {
    double** vectors = Prologue(self, args);
    if (vectors == NULL)
        return None();
        vectors = jacobi(vectors, global_rows);
    return Epilogue(vectors, global_rows, global_rows);
}


static PyMethodDef capiMethods[] = {
        {"fit", (PyCFunction)kmeans_capi, METH_VARARGS, PyDoc_STR(" ")},
        {"spk", (PyCFunction)spk_capi, METH_VARARGS, PyDoc_STR(" ")},
        {"wam", (PyCFunction)wam_capi, METH_VARARGS, PyDoc_STR(" ")},
        {"ddg", (PyCFunction)ddg_capi, METH_VARARGS, PyDoc_STR(" ")},
        {"lnorm", (PyCFunction)lnorm_capi, METH_VARARGS, PyDoc_STR(" ")},
        {"jacobi", (PyCFunction)jacobi_capi, METH_VARARGS, PyDoc_STR(" ")},
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef ={
        PyModuleDef_HEAD_INIT,
        "spkmeansmodule", /* name of module */
        NULL,
        -1,
        capiMethods};

PyMODINIT_FUNC
PyInit_spkmeansmodule(void)
{
    return PyModule_Create(&moduledef);
}

