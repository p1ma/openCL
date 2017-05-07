#pragma OPENCL EXTENSION cl_khr_global_int64_base_atomics : enable

__kernel void compact(
  __global const float *A,
  __global int  *coords,
  __global float *values,
  __global int *ctr
 )
{
  int x, y, grp, gid, num;
  gid = get_global_id( 0 );
  grp = get_local_size( 0 );
  num = get_num_groups(0) - 1;
  x = gid % grp;
  y = gid / grp;

  if ( A[ gid ] > 0.0f ) {
    int val =  ctr[gid];
    coords[(2 * val)] = x;
    coords[(2 * val) + 1] = y;
    values[ val ] = A[ gid ];
  }
}

__kernel void expand(
  __global float       *A,
  __global const int   *coords,
  __global const float *values
 )
{
 int x, grp, glb, index;
 grp = get_local_size( 0 );
 glb = get_global_size( 0 );
 x = get_global_id( 0 );
 index = 2 * x;
 
 if ( (index + 1) < glb) {
    A[coords[index] + (coords[index + 1]) * grp] = values[x];
  }
}
