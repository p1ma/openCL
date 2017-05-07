
__kernel void kernel1( __global const int *table,
 __global int *wsums,
 const int N,
 const int K)
{
    int wsum = 0;
    int i = get_global_id(0);
        for(int j = -K; j <= K ; j++) {
        /*
            accessed case : from [(0 + -K) % N] to [(N + (K-1) % N]
            we our value we got :
            [1048544] to [31]

        */
        int index = ((i + j) % N);
        /*
            Could have done padding to deal with
            these issue. table would be N + 2k sized.
        */
        if ( index <= 0 ) {
            index = 0;
        }
        /*
            2. when i is equals to 0
            The same location at 0 seems to be read
            31 times, because it enters the if conditional
            then when i is equals to 1 , we read 30 times the
            same case.
            it works till i is < 33
            so it about : (32*33) / 2

            the rest of the cases seems to be accessed 65 times every calls.
        */
            wsum = wsum + table[index];
        }
        wsums[i] = wsum;
}

__kernel void kernel12( __global const int *table,
 __global int *wsums,
 const int N,
 const int K)
{
    int wsum = 0;
    int i = get_global_id(0);

    // PRE FETCH
    prefetch(table, N);
        for(int j = -K; j <= K ; j++) {
        /*
            accessed case : from [(0 + -K) % N] to [(N + (K-1) % N]
            we our value we got :
            [1048544] to [31]

        */
        int index = ((i + j) % N);
        if ( index <= 0) {
            index = 0;
        }
        /*
            2. when i is equals to 0
            The same location at 0 seems to be read
            31 times, because it enters the if conditional
            then when i is equals to 1 , we read 30 times the
            same case.
            it works till i is < 33
            so it about : (32*33) / 2

            the rest of the cases seems to be accessed 65 times every call
        */
            wsum = wsum + table[index];
        }
        wsums[i] = wsum;
}

