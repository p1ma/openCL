// ----------------------------------------------------------

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

// ----------------------------------------------------------

#include "clutils.h"

int const N = 5; // in case of multiple kernel (1 << 7);
int const G = 5; // if 1 kernel then 1 group
int const maxRow = N; // max row for Pascal triangle

// ----------------------------------------------------------

/*
	Triangle de Pascal
*/
int main(int argc, char **argv)
{
  const char *clu_File = SRC_PATH "ex1.cl";

  // init OpenCL
  cluInit();

  // load program file
  cl::Program *prg = cluLoadProgram(clu_File);

  // allocate memory
    int* pascal = new int[maxRow * maxRow];

    // initialization
    for(int i = 0 ; i < maxRow * maxRow ; i++) {
            pascal[i] = 0;
    }

    // initialise la premiere case
    pascal[0] = 1;

    // table in
  cl::Buffer gpuTbl(
    *clu_Context,
    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
    sizeof(int)* maxRow * maxRow,
                    pascal);

// load kernel
cl::Program *p = cluLoadProgram( clu_File );
cl::Kernel  *k = cluLoadKernel ( p, "pascal" );

// set args
k->setArg(0, gpuTbl);

/*
    On appel plusieurs fois le kernel, selon le nombre de lignes
    qu'on veut developper
*/
cl_int err;
int j = 1;
for(int i = 0 ; i < maxRow ; i++) {
    k->setArg(1, j);
    err = clu_Queue->enqueueNDRangeKernel( *k, cl::NullRange, cl::NDRange(N), cl::NDRange(G) );
    cluCheckError( err, "enqueueNDRangeKernel" );
    j++;
}

// on charge les résultats dans le tableau pascal
clu_Queue->enqueueReadBuffer( gpuTbl, false, 0, sizeof(int) * maxRow * maxRow, pascal );
clu_Queue->finish();

for(int i = 0 ; i < maxRow ; i++) {
        for(int j = 0 ; j < (i + 1) ; j++) {
        cerr << pascal[(i * maxRow) + j] << " ";
    }
    cerr << "" << endl;
}

 // on libere la mémoire
    delete[] pascal;
    return (0);
}

// ----------------------------------------------------------
