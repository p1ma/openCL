
__kernel void kernelSansCoalescence( __global int* table,
int N )
{

    int id = get_global_id(0) ;

    int i = (49999 * id) % N;
    table[id] =  table[i - 1];
    i = (49999 * id) % N;
    table[id + 1] =  table[i];
    i = (49999 * id) % N;
    table[id + 2] =  table[i];
    i = (49999 * id) % N;
    table[id + 3] =  table[i];
    i = (49999 * id) % N;
    table[id + 4] =  table[i];
}

__kernel void kernelAvecCoalescence( __global int* table,
int N )
{
	int i = get_global_id(0) % N;
	int j = get_global_id(0) / N;
    int id  = get_global_id( 0 );
    int f = N-1-id;
    table[id] = table[i*N + j];
    table[id + 1] = table[i*N + j];
    table[id + 2] = table[i*N + j];
    table[id + 3] = table[i*N + j];
    table[id + 4] = table[i*N + j];
}
