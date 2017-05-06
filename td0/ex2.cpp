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

int N = (1 << 5);
int G = 4;


// ----------------------------------------------------------

int main(int argc,char **argv)
{
  const char *clu_File = SRC_PATH "ex2.cl";

  // init OpenCL
  cluInit();

  // load program file
  cl::Program *prg = cluLoadProgram(clu_File);

  // Je vais essayer de faire G acces a ce tab grace aux kernels
  int *tab1 = new int[N];
    for(int i = 0 ; i < N ; i++) {
        tab1[i] = (i+1);
    }

// table in
	cl::Buffer gpuTbl(
    *clu_Context,
    CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
    sizeof(int) * N,
    tab1);

// load 1rst kernel (non coalesced)
	cl::Program *p = cluLoadProgram( clu_File );

	// 1ND FIRST
	cl_int err;
	cl::Event ev;
	cl::Kernel *k = cluLoadKernel ( p, "kernelAvecCoalescence" ); // change arg

	k->setArg(0, gpuTbl);
	k->setArg(1, N);

	err = clu_Queue->enqueueNDRangeKernel( *k, cl::NullRange, cl::NDRange(N), cl::NDRange(G), NULL, &ev  );
	cluCheckError( err, "enqueueNDRangeKernel" );
	ev.wait();
	cluDisplayEventMilliseconds("[COALESCED] kernel time  ",ev);

	// 2ND KERNEL
	k = cluLoadKernel ( p, "kernelSansCoalescence" ); // change arg

	int *tab2 = new int[N];
    for(int i = 0 ; i < N ; i++) {
        tab2[i] = (i+1);
    }

	cl::Buffer gpuTbl2(
    *clu_Context,
    CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
    sizeof(int) * N,
    tab2);

	k->setArg(0, gpuTbl2);
	k->setArg(1, N);

	err = clu_Queue->enqueueNDRangeKernel( *k, cl::NullRange, cl::NDRange(N), cl::NDRange(G), NULL, &ev  );
	cluCheckError( err, "enqueueNDRangeKernel" );
	ev.wait();
	cluDisplayEventMilliseconds("[without COALESCED] kernel time  ",ev);

	delete[] tab2;
	delete[] tab1;
  return (0);
}

// ----------------------------------------------------------
