#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define S_LEN 512
#define N 1000

#define PEN_INS -2
#define PEN_DEL -2
#define PEN_MATCH 1
#define PEN_MISMATCH -1

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

double get_time() // function that returns the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__device__ int max4(int n1, int n2, int n3, int n4)
{
	int tmp1, tmp2;
	tmp1 = n1 > n2 ? n1 : n2;
	tmp2 = n3 > n4 ? n3 : n4;
	tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
	return tmp1;
}

__device__ void backtrace(char *simple_rev_cigar, char **dir_mat, int i, int j, int max_cigar_len)
{
	int n;
	for (n = 0; n < max_cigar_len && dir_mat[i][j] != 0; n++)
	{
		int dir = dir_mat[i][j];
		if (dir == 1 || dir == 2)
		{
			i--;
			j--;
		}
		else if (dir == 3)
			i--;
		else if (dir == 4)
			j--;

		simple_rev_cigar[n] = dir;
	}
}


__global__ void GPUfn(char ** query, char ** reference, int ** sc_mat, char ** dir_mat, int * res, char ** simple_rev_cigar) {    
	for (int n = 0; n < N; n++)
	{
		int max = PEN_INS; // in sw all scores of the alignment are >= 0, so this will be for sure changed
		int maxi, maxj;
		// initialize the scoring matrix and direction matrix to 0
		for (int i = 0; i < S_LEN + 1; i++)
		{
			for (int j = 0; j < S_LEN + 1; j++)
			{
				sc_mat[i][j] = 0;
				dir_mat[i][j] = 0;
			}
		}
		// compute the alignment
		for (int i = 1; i < S_LEN; i++)
		{
			for (int j = 1; j < S_LEN; j++)
			{
				// penalities (initialized above)
				// 	ins = -2
				// 	del = -2
				// 	match = 1
				// 	mismatch = -1 

				// compare the sequences characters
				int comparison = (query[n][i - 1] == reference[n][j - 1]) ? PEN_MATCH : PEN_MISMATCH;
				// compute the cell knowing the comparison result
				int tmp = max4(sc_mat[i - 1][j - 1] + comparison, sc_mat[i - 1][j] + PEN_DEL, sc_mat[i][j - 1] + PEN_INS, 0);
				char dir;

				// directions
				// 	1 -> up + left
				// 	2 -> up + left
				// 	3 -> left
				// 	4 -> up
				//  0 -> none.

				if (tmp == (sc_mat[i - 1][j - 1] + comparison))
					dir = comparison == PEN_MATCH ? 1 : 2;
				else if (tmp == (sc_mat[i - 1][j] + PEN_DEL))
					dir = 3;
				else if (tmp == (sc_mat[i][j - 1] + PEN_INS))
					dir = 4;
				else
					dir = 0;

				dir_mat[i][j] = dir;
				sc_mat[i][j] = tmp;

				if (tmp > max)
				{
					max = tmp;
					maxi = i;
					maxj = j;
				}
			}
		}
		res[n] = sc_mat[maxi][maxj];
		backtrace(simple_rev_cigar[n], dir_mat, maxi, maxj, S_LEN * 2);
	}
}

int main(int argc, char * argv[]) {
	srand(time(NULL)); 
    char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};
    double time_start, time_stop;

    // Host memory allocation and initialization for sequences (randomly generated)
    char **h_query = (char **)malloc(N * sizeof(char *));
	for (int i = 0; i < N; i++)
		h_query[i] = (char *)malloc(S_LEN * sizeof(char));

	char **h_reference = (char **)malloc(N * sizeof(char *));
	for (int i = 0; i < N; i++)
		h_reference[i] = (char *)malloc(S_LEN * sizeof(char));

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < S_LEN; j++)
		{
			h_query[i][j] = alphabet[rand() % 5];
			h_reference[i][j] = alphabet[rand() % 5];
		}
	}

    // Device memory allocation and transfer of input data
    char **d_query;
    char **d_query_ptrs = (char **)malloc(N * sizeof(char *));    
    cudaMalloc((void***)&d_query,  N * sizeof(char *));
    for(int i=0; i<N; i++) {
        cudaMalloc((void**) &(d_query_ptrs[i]), S_LEN*sizeof(char));
        cudaMemcpy (d_query_ptrs[i], h_query[i], S_LEN*sizeof(char), cudaMemcpyHostToDevice);
    }
    cudaMemcpy (d_query, d_query_ptrs, N*sizeof(char *), cudaMemcpyHostToDevice);

    char **d_reference;
    char **d_reference_ptrs = (char **)malloc(N * sizeof(char *));    
    cudaMalloc((void***)&d_reference,  N * sizeof(char *));
    for(int i=0; i<N; i++) {
        cudaMalloc((void**) &(d_reference_ptrs[i]), S_LEN*sizeof(char));
        cudaMemcpy (d_reference_ptrs[i], h_reference[i], S_LEN*sizeof(char), cudaMemcpyHostToDevice);
    }
    cudaMemcpy (d_reference, d_reference_ptrs, N*sizeof(char *), cudaMemcpyHostToDevice);

    int **d_sc_mat;
    int **d_sc_mat_ptrs = (int **)malloc((S_LEN+1) * sizeof(int *));    
    cudaMalloc((void***)&d_sc_mat, (S_LEN+1) * sizeof(int *));
    for(int i=0; i<(S_LEN+1); i++) {
        cudaMalloc((void**) &(d_sc_mat_ptrs[i]), (S_LEN+1)*sizeof(int));
    }
    cudaMemcpy (d_sc_mat, d_sc_mat_ptrs, (S_LEN+1)*sizeof(int *), cudaMemcpyHostToDevice);

    char **d_dir_mat;
    char **d_dir_mat_ptrs = (char **)malloc((S_LEN+1) * sizeof(char *));    
    cudaMalloc((void***)&d_dir_mat, (S_LEN+1) * sizeof(char *));
    for(int i=0; i<(S_LEN+1); i++) {
        cudaMalloc((void**) &(d_dir_mat_ptrs[i]), (S_LEN+1)*sizeof(char));
    }
    cudaMemcpy (d_dir_mat, d_dir_mat_ptrs, (S_LEN+1)*sizeof(char *), cudaMemcpyHostToDevice);

    int * d_res;
    cudaMalloc((void**) &d_res, N*sizeof(int));

    char **d_simple_rev_cigar;
    char **d_simple_rev_cigar_ptrs = (char **)malloc(N * sizeof(char *));    
    cudaMalloc((void***)&d_simple_rev_cigar, N * sizeof(char *));
    for(int i=0; i<N; i++) {
        cudaMalloc((void**) &(d_simple_rev_cigar_ptrs[i]), S_LEN * 2 * sizeof(char));
    }
    cudaMemcpy (d_simple_rev_cigar, d_simple_rev_cigar_ptrs, N*sizeof(char *), cudaMemcpyHostToDevice);


    // Blocks and threads schema for GPU execution
    dim3 blocksPerGrid(N, 1, 1);
    dim3 threadsPerBlock((1), 1, 1);

    // Execution on GPU started
    time_start = get_time();

    GPUfn<<<blocksPerGrid, threadsPerBlock>>>(d_query, d_reference, d_sc_mat, d_dir_mat, d_res, d_simple_rev_cigar);
    CHECK_KERNELCALL();
    cudaDeviceSynchronize();

    // Execution on GPU ended
    time_stop = get_time();

    printf("Execution time: %.10f\n", time_stop-time_start);

    return 0;
}