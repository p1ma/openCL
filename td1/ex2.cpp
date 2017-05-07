// ----------------------------------------------------------

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

// ----------------------------------------------------------

#include "clutils.h"

const int N = (1 << (12));
const int G = (1 << (5));

// ----------------------------------------------------------

int main(int argc,char **argv)
{
    const char *clu_File = SRC_PATH "ex2.cl";

    // init OpenCL
    cluInit();

    // load program file
    cl::Program *prg = cluLoadProgram(clu_File);

    int *table = new int[N];
    int sum = 0;
    for(int i = 0 ; i < N ; i++) {
        table[i] = rand() % 100;
        sum += table[i];
    }
    	cerr << "sum : " << sum << endl;

    // SUM v1
    int *resultat = new int[1];
    *resultat = 0;

    // table input 1
    cl::Buffer gpu_table(
                *clu_Context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(int) * N,
                table);

    // table input 2
    cl::Buffer gpu_resultat(
                *clu_Context,
                CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
                sizeof(int),
                resultat);

    // load kernel
    cl::Kernel  *k = cluLoadKernel ( prg, "sum1" );
    // sets args
    k->setArg(0, gpu_table);
    k->setArg(1, gpu_resultat);

    cl_int err;
    cl::Event ev;
    err = clu_Queue->enqueueNDRangeKernel( *k, cl::NullRange, cl::NDRange(N), cl::NDRange(G), NULL, &ev );
    ev.wait();
    cluDisplayEventMilliseconds("(SUM 1) Time spent : ", ev);

    clu_Queue->enqueueReadBuffer( gpu_resultat, false, 0, sizeof(int), resultat );
    clu_Queue->finish();

    if (*resultat != sum) {
        cerr << "(SUM 1) PROBLEM" << endl;
    }

    // SUM v2
    k = cluLoadKernel ( prg, "sum2" );

    *resultat = 0;
    // table input 1
    cl::Buffer gpu_resultat_bis(
                *clu_Context,
                CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
                sizeof(int),
                resultat);

    // sets args
    k->setArg(0, gpu_table);
    k->setArg(1, gpu_resultat_bis);

    const int K = sqrt(N);
    const int M = K / 2;
    err = clu_Queue->enqueueNDRangeKernel( *k, cl::NullRange, cl::NDRange(K), cl::NDRange(M), NULL, &ev );
    ev.wait();
    cluDisplayEventMilliseconds("(SUM 2) Time spent : ", ev);

    clu_Queue->enqueueReadBuffer( gpu_resultat_bis, false, 0, sizeof(int), resultat );
    clu_Queue->finish();

    if (*resultat != sum) {
        cerr << "(SUM 2) PROBLEM" << endl;
    }

    // SUM v3
    k = cluLoadKernel ( prg, "sum3" );
    *resultat = 0;
    
    int *subC = new int[M];
    for(int i = 0 ; i < M ; i++) { subC[i] = 0; }

    // table input 1
    cl::Buffer gpu_resultat_bis2(
                *clu_Context,
                CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
                sizeof(int),
                resultat);

    cl::Buffer gpu_subCounter(
                *clu_Context,
                CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
                sizeof(int) * M,
                subC);

    // sets args
    k->setArg(0, gpu_table);
    k->setArg(1, gpu_subCounter);
    k->setArg(2, gpu_resultat_bis2);

    err = clu_Queue->enqueueNDRangeKernel( *k, cl::NullRange, cl::NDRange(K), cl::NDRange(M), NULL, &ev );
    ev.wait();
    cluDisplayEventMilliseconds("(SUM 3) Time spent : ", ev);

    clu_Queue->enqueueReadBuffer( gpu_resultat_bis2, false, 0, sizeof(int), resultat );
    clu_Queue->finish();

    if (*resultat != sum) {
        cerr << "(SUM 3) PROBLEM" << endl;
    }

    return (0);
}

// ----------------------------------------------------------
