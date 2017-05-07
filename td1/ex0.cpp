// ----------------------------------------------------------

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

// ----------------------------------------------------------

#include "clutils.h"
const int N = ( 1 << 20);
const int K = 32;

// ----------------------------------------------------------
/*
    Answer of question 2 is inside the kernel's code
*/
int main(int argc,char **argv)
{
    const char *clu_File = SRC_PATH "ex0.cl";

    // init OpenCL
    cluInit();

    // load program file
    cl::Program *prg = cluLoadProgram(clu_File);

    // arrays initialization
    int * table = new int[N];
    // I put values 1
    for(int i = 0 ; i < N ; i++) {
        table[i] = 1;
    }
    int * wsums = new int[N];

    // table input
    cl::Buffer gpuTblIn(
                *clu_Context,
                CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                sizeof(int)* N,
                table);

    // table output
    cl::Buffer gpuTblOut(
                *clu_Context,
                CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
                sizeof(int)* N,
                wsums);

    // load kernel
    cl::Kernel  *k = cluLoadKernel ( prg, "kernel1" );

    // we set the args
    k->setArg(0, gpuTblIn);
    k->setArg(1, gpuTblOut);
    k->setArg(2, N);
    k->setArg(3, K);

    // we launch the N kernels
    cl_int err;
    cl::Event ev;
    err = clu_Queue->enqueueNDRangeKernel( *k, cl::NullRange, cl::NDRange(N), cl::NDRange(K), NULL, &ev );
    cluCheckError( err, "enqueueNDRangeKernel" );
    ev.wait();

    /*
    3.
    (time without prefetch) -> ~17.8x ms
    (time with prefetch) -> ~ 17.7x ms

*/
    cluDisplayEventMilliseconds("execution time",ev);

    // load results
    clu_Queue->enqueueReadBuffer( gpuTblOut, false, 0, sizeof(int) * N , wsums );
    clu_Queue->finish();

    // K first one
    cerr << "K FIRSTS" << endl;
    for(int i = 0 ; i < K ; i++) {
        cerr << wsums[i] << endl;
    }
    //K last one
    cerr << " K LASTS" << endl;
    for(int i = N-1 ; i > N-K ; i--) {
        cerr << wsums[i] << endl;
    }
    return (0);
}

// ----------------------------------------------------------
