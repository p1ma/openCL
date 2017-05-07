#pragma OPENCL EXTENSION cl_khr_global_int64_base_atomics : enable

__kernel void sum1(
__global const int *table,
__global int * res
  ) {
  int x = get_global_id( 0 );
  int v = table[ x ];
  atomic_add(&res[0], v);
}

__kernel void sum2(
__global const int *table,
__global int * res
  ) {
  int x = get_global_id( 0 );
  int grp = get_global_size( 0 );
  int v = 0;

  for(int i = 0 ; i < grp ; i++) {
    v += table[ (x * grp) + i ];
  }
  atomic_add(&res[0], v);
}

__kernel void sum3(
__global const int *table,
__global int *subCounters,
__global int *gCounter
  ) {
  int x = get_global_id( 0 );
  int grp = get_global_size( 0 );
  int num = get_num_groups(0) - 1;
  int v = 0;

  for(int i = 0 ; i < grp ; i++) {
    v = table[ (x * grp) + i ];
    atomic_add(&subCounters[num], v);
  }

  if ( ((x + 1) % grp) == 0 ) atomic_add(&gCounter[0], subCounters[num]);
}