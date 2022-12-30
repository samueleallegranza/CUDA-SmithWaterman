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

#define VAL 0
#define J   1

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

__device__ int IDX_MTX(int block, int dim, int i, int j) {
	return (block*dim*dim)+(i*dim+j);
}

__device__ int IDX_ARR(int block, int dim, int i) {
	return (block*dim)+i;
}

__device__ int max4(int n1, int n2, int n3, int n4)
{
	int tmp1, tmp2;
	tmp1 = n1 > n2 ? n1 : n2;
	tmp2 = n3 > n4 ? n3 : n4;
	tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
	return tmp1;
}

__device__ void backtrace(char *simple_rev_cigar, char *dir_mat, int i, int j, int max_cigar_len, int block)
{
	int n;
	for (n = 0; n < max_cigar_len && dir_mat[IDX_MTX(block, (S_LEN+1), i, j)] != 0; n++)
	{
		int dir = dir_mat[IDX_MTX(block, (S_LEN+1), i, j)];
		if (dir == 1 || dir == 2)
		{
			i--;
			j--;
		}
		else if (dir == 3)
			i--;
		else if (dir == 4)
			j--;

		simple_rev_cigar[IDX_ARR(block, S_LEN*2, n)] = dir;
	}
}

__device__ int setDirScore(char * query, char * reference, int * sc_mat, char * dir_mat, int x, int y, int block) {
	// penalities (initialized above)
	// 	ins = -2
	// 	del = -2
	// 	match = 1
	// 	mismatch = -1 

	// compare the sequences characters
	int comparison = (query[IDX_ARR(block, S_LEN, x-1)] == reference[IDX_ARR(block, S_LEN, y-1)]) ? PEN_MATCH : PEN_MISMATCH;
	// compute the cell knowing the comparison resultS
	int score = max4(sc_mat[IDX_MTX(block, S_LEN+1, x-1, y-1)] + comparison, sc_mat[IDX_MTX(block, S_LEN+1, x-1, y)] + PEN_DEL, sc_mat[IDX_MTX(block, S_LEN+1, x, y-1)] + PEN_INS, 0);

	// directions
	// 	1 -> up + left
	// 	2 -> up + left
	// 	3 -> left
	// 	4 -> up
	//  0 -> none.
	char dir;

	if (score == (sc_mat[IDX_MTX(block, S_LEN+1, x-1, y-1)] + comparison))
		dir = comparison == PEN_MATCH ? 1 : 2;
	else if (score == (sc_mat[IDX_MTX(block, S_LEN+1, x-1, y)] + PEN_DEL))
		dir = 3;
	else if (score == (sc_mat[IDX_MTX(block, S_LEN+1, x, y-1)] + PEN_INS))
		dir = 4;
	else
		dir = 0;

	dir_mat[IDX_MTX(block, S_LEN+1, x, y)] = dir;
	sc_mat[IDX_MTX(block, S_LEN+1, x, y)] = score;

	return score;
}

__global__ void sw_GPU(char * query, char * reference, int * sc_mat, char * dir_mat, int * res, char * simple_rev_cigar) {    
    int n = blockIdx.x;

    int i, progress;
    int score;
    int max, maxi, maxj;

    // clock_t start, end;

    __shared__ int max_supp[S_LEN][2];

    for(i=0; i<S_LEN; i++) {
        sc_mat[IDX_MTX(n, (S_LEN+1), i, threadIdx.x)] = 0;
        dir_mat[IDX_MTX(n, (S_LEN+1), i, threadIdx.x)] = 0;
    }
    // Additional instruction for the remaining row
    if(threadIdx.x == S_LEN-1) {
        sc_mat[IDX_MTX(n, (S_LEN+1), i, S_LEN)] = 0;
        dir_mat[IDX_MTX(n, (S_LEN+1), i, S_LEN)] = 0;
    }

    progress = 1;
    for(i=1; i<S_LEN*2; i++) {
        if(threadIdx.x < i && progress <= S_LEN) {
            score = setDirScore(query, reference, sc_mat, dir_mat, threadIdx.x+1, progress, n);
            progress++;
        }
        __syncthreads();
    }


    max_supp[threadIdx.x][VAL] = sc_mat[IDX_MTX(n, (S_LEN+1), threadIdx.x, 1)];
    max_supp[threadIdx.x][J] = 1;
    for(i=2; i<S_LEN; i++) {
        if(sc_mat[IDX_MTX(n, (S_LEN+1), threadIdx.x, i)] > max_supp[threadIdx.x][VAL]) {
            max_supp[threadIdx.x][VAL] = sc_mat[IDX_MTX(n, (S_LEN+1), threadIdx.x, i)];
            max_supp[threadIdx.x][J] = i;
        }
    }
        
    __syncthreads();        
    if(threadIdx.x == 0) {
        max = max_supp[0][VAL];
        maxi = threadIdx.x;
        maxj = max_supp[0][J];
        for(i=1; i<S_LEN; i++) {
            if(max_supp[i][VAL] > max) {
                max = max_supp[i][VAL];
                maxi = i;
                maxj = max_supp[i][J];
            }
        }    

        res[n] = sc_mat[IDX_MTX(n, (S_LEN+1), maxi, maxj)];
        backtrace(simple_rev_cigar, dir_mat, maxi, maxj, S_LEN*2, n);

        // if(blockIdx.x == 0) {
        //     printf("Score matrix\t\tDirection matrix\n");
        //     for(i=0; i<S_LEN; i++) {
        //         for(j=0; j<S_LEN+1; j++) {
        //             printf("%d ", sc_mat[i][j]);
        //         }
        //         printf("\t");
        //         for(j=0; j<S_LEN+1; j++) {
        //             printf("%d ", dir_mat[i][j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("max = %d, maxi = %d, maxj = %d\n", max, maxi, maxj);
        // }
    }
}


__host__ int max4_CPU(int n1, int n2, int n3, int n4)
{
	int tmp1, tmp2;
	tmp1 = n1 > n2 ? n1 : n2;
	tmp2 = n3 > n4 ? n3 : n4;
	tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
	return tmp1;
}

__host__ void backtrace_CPU(char *simple_rev_cigar, char **dir_mat, int i, int j, int max_cigar_len)
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

__host__ void sw_CPU(char ** query, char ** reference, int ** sc_mat, char ** dir_mat, int * res, char ** simple_rev_cigar) {
	int ins = -2, del = -2, match = 1, mismatch = -1; // penalties
	
	for (int n = 0; n < N; n++)
	{
		int max = ins; // in sw all scores of the alignment are >= 0, so this will be for sure changed
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
				// compare the sequences characters
				int comparison = (query[n][i - 1] == reference[n][j - 1]) ? match : mismatch;
				// compute the cell knowing the comparison result
				int tmp = max4_CPU(sc_mat[i - 1][j - 1] + comparison, sc_mat[i - 1][j] + del, sc_mat[i][j - 1] + ins, 0);
				char dir;

				if (tmp == (sc_mat[i - 1][j - 1] + comparison))
					dir = comparison == match ? 1 : 2;
				else if (tmp == (sc_mat[i - 1][j] + del))
					dir = 3;
				else if (tmp == (sc_mat[i][j - 1] + ins))
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
		backtrace_CPU(simple_rev_cigar[n], dir_mat, maxi, maxj, S_LEN * 2);

        // if(n == 0) {
        //     printf("Score matrix\t\tDirection matrix\n");
        //     for(int i=0; i<S_LEN; i++) {
        //         for(int j=0; j<S_LEN+1; j++) {
        //             printf("%d ", sc_mat[i][j]);
        //         }
        //         printf("\t");
        //         for(int j=0; j<S_LEN+1; j++) {
        //             printf("%d ", dir_mat[i][j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("max = %d, maxi = %d, maxj = %d\n", max, maxi, maxj);
        // }
    }
}

int main(int argc, char * argv[]) {
	srand(time(NULL)); 
    char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};
    double time_start, time_stop;
	int rand_query, rand_reference;

    // Host memory
    char **h_query = (char **)malloc(N * sizeof(char *));
	for (int i = 0; i < N; i++)
		h_query[i] = (char *)malloc(S_LEN * sizeof(char));
	char **h_reference = (char **)malloc(N * sizeof(char *));
	for (int i = 0; i < N; i++)
		h_reference[i] = (char *)malloc(S_LEN * sizeof(char));

	int **h_sc_mat = (int **)malloc((S_LEN + 1) * sizeof(int *));
	for (int i = 0; i < (S_LEN + 1); i++)
		h_sc_mat[i] = (int *)malloc((S_LEN + 1) * sizeof(int));
	char **h_dir_mat = (char **)malloc((S_LEN + 1) * sizeof(char *));
	for (int i = 0; i < (S_LEN + 1); i++)
		h_dir_mat[i] = (char *)malloc((S_LEN + 1) * sizeof(char));

	int *h_res = (int *)malloc(N * sizeof(int));
	char **h_simple_rev_cigar = (char **)malloc(N * sizeof(char *));
	for (int i = 0; i < N; i++)
		h_simple_rev_cigar[i] = (char *)malloc(S_LEN * 2 * sizeof(char));

    // Device memory
	char *d_query, *d_reference, *d_dir_mat, *d_simple_rev_cigar;
	int *d_sc_mat, *d_res;
    char *d_query_supp = (char *) malloc(N*S_LEN * sizeof(char));
    char *d_reference_supp = (char *) malloc(N*S_LEN * sizeof(char));    

	// Initialization of query and reference (randomly generated)
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < S_LEN; j++)
		{
			rand_query = rand() % 5;
			rand_reference = rand() % 5;
			h_query[i][j] = alphabet[rand_query];
			h_reference[i][j] = alphabet[rand_reference];
			d_query_supp[i*S_LEN+j] = alphabet[rand_query];
			d_reference_supp[i*S_LEN+j] = alphabet[rand_reference];
		}
	}

	// Device memory allocation
	CHECK(cudaMalloc(&d_query, 				N*S_LEN * sizeof(char)));
	CHECK(cudaMalloc(&d_reference, 			N*S_LEN * sizeof(char)));
	CHECK(cudaMalloc(&d_sc_mat, 			N*(S_LEN+1)*(S_LEN+1) * sizeof(int)));
	CHECK(cudaMalloc(&d_dir_mat, 			N*(S_LEN+1)*(S_LEN+1) * sizeof(char)));
	CHECK(cudaMalloc(&d_simple_rev_cigar, 	N*2*(S_LEN) * sizeof(char)));
	CHECK(cudaMalloc(&d_res, 				N * sizeof(int)));

	CHECK(cudaMemcpy(d_query, d_query_supp,			N*S_LEN * sizeof(char), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_reference, d_reference_supp, N*S_LEN * sizeof(char), cudaMemcpyHostToDevice));

    // Blocks and threads schema for GPU execution
    dim3 blocksPerGrid(N, 1, 1);
    dim3 threadsPerBlock(S_LEN, 1, 1);

    // Execution on GPU started
    time_start = get_time();

    sw_GPU<<<blocksPerGrid, threadsPerBlock>>>(d_query, d_reference, d_sc_mat, d_dir_mat, d_res, d_simple_rev_cigar);
    CHECK_KERNELCALL();
	// The following function adds A LOT of latency! Don't use it for benchmarking purposes
    // cudaDeviceSynchronize();

    // Execution on GPU ended
    time_stop = get_time();
    printf("GPU Execution time: %.10f\n", time_stop-time_start);


	// TODO: Error checking to verify that results are correct
	
	// Transfer data to host
	// int * gpu_res = (int *) malloc(sizeof(int)*N);
	// int gpu_res[N];
	// char gpu_simple_rev_cigar[N][S_LEN*2];
	// char * tmp_pointer;

	// cudaMemcpy((void *) gpu_res, d_res, sizeof(int)*N, cudaMemcpyDeviceToHost);
	// for(int i=0; i<N; i++) {
	// 	tmp_pointer = d_simple_rev_cigar_ptrs[i];
	// 	cudaMemcpy((void *) gpu_simple_rev_cigar[i], tmp_pointer, sizeof(char)*S_LEN*2, cudaMemcpyDeviceToHost);
	// }

    // // Execution on CPU started
    // time_start = get_time();

	// sw_CPU(h_query, h_reference, h_sc_mat, h_dir_mat, h_res, h_simple_rev_cigar);
    
	// // Execution on CPU ended
    // time_stop = get_time();
    // printf("CPU Execution time: %.10f [!]\n", time_stop-time_start);

	// // Check if results are consistent
	// // "Results" validation
	// int ok, i, j;
	// ok = 1;
	// for(i=0; (i<N && ok); i++) {
	// 	ok = gpu_res[i] == h_res[i];
	// }

	// if(ok)
	// 	printf("[OK]\t'results' is consistent\n");
	// else
	// 	printf("[ERR]\t'results' is inconsistent\n");

	// ok = 1;
	// for(i=0; (i<N && ok); i++) {
	// 	for(j=0; (j<S_LEN*2 && ok); j++) {
	// 		ok = gpu_simple_rev_cigar[i][j] == h_simple_rev_cigar[i][j];
	// 	}
	// }
	// i--;
	// if(ok)
	// 	printf("[OK]\t'rev_cigar' is consistent\n");
	// else{
	// 	printf("[ERR]\t'rev_cigar' is inconsistent [err on: %d]\n", i);
	// }


	// Deallocation of memory
	CHECK(cudaFree(d_query));
    CHECK(cudaFree(d_reference));
    CHECK(cudaFree(d_sc_mat));
    CHECK(cudaFree(d_dir_mat));
    CHECK(cudaFree(d_res));
    CHECK(cudaFree(d_simple_rev_cigar));

	free(d_query_supp);
	free(d_reference_supp);

	free(h_query);
	free(h_reference);
	free(h_sc_mat);
	free(h_dir_mat);
	free(h_res);
	free(h_simple_rev_cigar);

    return 0;
}