
__kernel void pascal( __global int *table,
int j )
{
	int i = get_global_id ( 0 );
	int grp = get_local_size( 0 );
	if (i <= 0) {
		table[(grp * j) + i] = table[(grp * (j - 1)) + i] + 0;
	} else {
		table[(grp * j) + i] = table[(grp * (j - 1)) + i] + table[(grp * (j - 1)) + (i - 1)];
	}
}
