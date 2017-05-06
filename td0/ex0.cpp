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

const int N =  ( 1 << (24) ); // 2^24
const int G =  32;
// ----------------------------------------------------------

int main(int argc,char **argv)
{
  // nom du kernel
  const char *clu_File = SRC_PATH "ex0.cl";

  // init OpenCL
  cluInit();

  // on load le programme
  cl::Program *prg = cluLoadProgram(clu_File);

  // Allocations des 3 tableaux (CPU).
  int *a = new int[N];
  int *b = new int[N];
  int *c = new int[N];

  // c[i] = a[i] + b[i]
long long start = cluCPUMilliseconds();
  for(int i = 0 ; i < N ; i++) {
    a[i] = i;
    b[i] = i;
    c[i] = a[i] + b[i];
  }

 long long end   = cluCPUMilliseconds();
 cerr << "[(CPU) Time spent]  " << (end-start) << " msecs." << endl;


 // Partie GPU

/*
  On prepare 3 tableaux pour le GPU
  2 en lectures seules
  1 en écriture
*/
 cl::Buffer gpuTbl1(
    *clu_Context,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    sizeof(int) * N,
    a);

// je charge b
cl::Buffer gpuTbl2(
    *clu_Context,
     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    sizeof(int) * N,
   b);

// je créé le tableau c_gpu pour charger le resultat GPU
int *c_gpu = new int[N];
cl::Buffer gpuTbl3(
    *clu_Context,
    CL_MEM_WRITE_ONLY,
    sizeof(int) * N,
    c_gpu);

// on charge le programme
cl::Program *p = cluLoadProgram( clu_File );

// on charge le kernel
cl::Kernel  *k = cluLoadKernel ( p, "somme" );

// on set les arguments du kernel
k->setArg(0, gpuTbl1 );
k->setArg(1, gpuTbl2 );
k->setArg(2, gpuTbl3 );

/*
    Je créé autant de threads que de cases du tableaux donc N threads
*/
cl_int err;
cl::Event ev;

err = clu_Queue->enqueueNDRangeKernel( *k, cl::NullRange, cl::NDRange(N), cl::NDRange(G), NULL, &ev );
cluCheckError( err, "enqueueNDRangeKernel" );

ev.wait();
cluDisplayEventMilliseconds("(GPU) Time spent",ev);

// On charge le résultat issu du GPU
int *res = new int[N];
clu_Queue->enqueueReadBuffer( gpuTbl3, false, 0, sizeof(int) * N, res );
clu_Queue->finish();

/*
    J'itere et je vérifie que les 2 tableaux sont identiques.
*/
for(int i = 0 ; i < N ; i++) {
    if (c[i] != res[i]) {
        cerr << "Problem ! Results are differentes." << endl;
        cerr << "index " << i << "incorrect." << endl;
        exit(1);
    }
}

/*
    TEMPS :
    2^24 : CPU en moyenne ~55 ms, GPU en moyenne ~1.11ms
    2^16 : CPU en moyenne ~0 ms, GPU en moyenne ~0.05ms
    2^8 : CPU en moyenne ~0 ms, GPU en moyenne ~0ms

*/

// on efface memoire
  delete[] (a);
  delete[] (b);
  delete[] (c);
  return (0);
}

// ----------------------------------------------------------
