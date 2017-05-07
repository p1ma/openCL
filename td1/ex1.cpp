// ----------------------------------------------------------

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

// ----------------------------------------------------------

#include "clutils.h"

// matrix's dimension
int const N = (1 << 10);
int const M = (1 << 5);
int const WI = M * N ; // workItem
int const WG = (M); // groupGroup

// ----------------------------------------------------------

/*
    Je viens de me rendre compte que j'ai commenté en anglais
    sur le précèdent exercice et que là j'ai commenté en francais,
    veuillez m'excuser pour le manque de concordance.

*/
int main(int argc,char **argv)
{
    const char *clu_File = SRC_PATH "ex1.cl";

    // init OpenCL
    cluInit();

    // load program file
    cl::Program *program = cluLoadProgram(clu_File);

    // matrix
    float *A = new float[N * M];
    float *A_bis = new float[N * M];

    // out arrays
    int *coords = new int[N * M];
    float *values = new float[N * M];
    int *index = new int[N * M];
    int nonZeroNumber = 0;
    for(int i = 0 ; i < N ; i++) {
        for(int j = 0 ; j < M ; j++) {
            if( (rand() % 100) < 25 ) {
                A[j + (i * M)] = rand() % 10 + 1;
                index[j + (i * M)] =  nonZeroNumber;
                nonZeroNumber++;
            } else {
                A[j + (i * M)] = .0;
                index[j + (i * M)] = .0;
            }
            coords[j + (i * M)] = .0;
            values[j + (i * M)] = .0;
            A_bis[j + (i * M)] = .0;
        }
    }

    // matrix A
    cl::Buffer gpu_A(
                *clu_Context,
                CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * M * N,
                A);

    cl::Buffer gpu_coords(
                *clu_Context,
                CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
                sizeof(int) * M * N,
                coords);

    cl::Buffer gpu_values(
                *clu_Context,
                CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * M * N,
                values);

    cl::Buffer gpu_index(
                *clu_Context,
                CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
                sizeof(int) * M * N,
                index);

    // load kernel
    cl::Kernel  *k = cluLoadKernel ( program, "compact" );

    // we set the args
    k->setArg(0, gpu_A);
    k->setArg(1, gpu_coords);
    k->setArg(2, gpu_values);
    k->setArg(3, gpu_index);

    cerr << "Number of threads : " << WI << endl;
    cerr << "Number of groups : " << WG << endl;
    cerr << "Threads per groups : " << (WI / WG) << endl << endl;
    cl_int err;
    cl::Event ev;
    err = clu_Queue->enqueueNDRangeKernel( *k, cl::NullRange, cl::NDRange(WI), cl::NDRange(WG), NULL, &ev );
    ev.wait();
    cluDisplayEventMilliseconds("(COMPACT) Time spent : ", ev);
    
    // load results
    clu_Queue->enqueueReadBuffer( gpu_coords, false, 0, sizeof(int) * N * M, coords );
    clu_Queue->enqueueReadBuffer( gpu_values, false, 0, sizeof(float) * N * M, values );
    clu_Queue->finish();

    /*   cerr << "Matrix A :" << endl;
    for (int i = 0 ; i < N ; i++) {
        for (int j = 0 ; j < M ; j++) {
            cerr << A[j + (i * M)] << " ";
        }
        cerr << "" << endl;
    }
    cerr << "Matrix coords :" << endl;
    for (int i = 0 ; i < N ; i++) {
        for (int j = 0 ; j < M ; j++) {
            cerr << coords[j + (i * M)] << " ";
        }
        cerr << "" << endl;
    }

    cerr << "Matrix values :" << endl;
    for (int i = 0 ; i < N ; i++) {
        for (int j = 0 ; j < M ; j++) {
            cerr << values[j + (i * M)] << " ";
        }
        cerr << "" << endl;
    }
        */
    k = cluLoadKernel ( program, "expand" );

    cl::Buffer gpu_A_bis(
                *clu_Context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * M * N,
                A_bis);

    cl::Buffer gpu_coords_bis(
                *clu_Context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(int) * M * N,
                coords);

    cl::Buffer gpu_values_bis(
                *clu_Context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * M * N,
                values);

    // we set the args
    k->setArg(0, gpu_A_bis);
    k->setArg(1, gpu_coords_bis);
    k->setArg(2, gpu_values_bis);

    cerr << "EXPAND..." << endl;
    err = clu_Queue->enqueueNDRangeKernel( *k, cl::NullRange, cl::NDRange(WI), cl::NDRange(WG), NULL, NULL );
    cerr << "EXPAND done" << endl;

    // load results
    clu_Queue->enqueueReadBuffer( gpu_A_bis, false, 0, sizeof(float) * N * M, A_bis );
    clu_Queue->finish();

    cerr << "Verification..." << endl;
    for (int i = 0 ; i < N ; i++) {
        for (int j = 0 ; j < M ; j++) {
            if (A[j + (i * M)] != A_bis[j + (i * M)]) {
                cerr << "PROBLEM" << endl;
            }
        }
    }

    cerr << "Verification done" << endl;
    return (0);
}

// ----------------------------------------------------------
