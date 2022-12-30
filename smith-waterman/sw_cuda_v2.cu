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

#define D 0
#define X 1
#define Y 2

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

typedef struct max_supp_s {
	int v[3];
	int i[3];
	int j[3];
} max_supp_t;

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

__device__ int setDirScore(char * query, char * reference, int ** sc_mat, char ** dir_mat, int x, int y) {
	// penalities (initialized above)
	// 	ins = -2
	// 	del = -2
	// 	match = 1
	// 	mismatch = -1 

	// compare the sequences characters
	int comparison = (query[x - 1] == reference[y - 1]) ? PEN_MATCH : PEN_MISMATCH;
	// compute the cell knowing the comparison resultS
	int score = max4(sc_mat[x - 1][y - 1] + comparison, sc_mat[x - 1][y] + PEN_DEL, sc_mat[x][y- 1] + PEN_INS, 0);

	// directions
	// 	1 -> up + left
	// 	2 -> up + left
	// 	3 -> left
	// 	4 -> up
	//  0 -> none.
	char dir;

	if (score == (sc_mat[x - 1][y - 1] + comparison))
		dir = comparison == PEN_MATCH ? 1 : 2;
	else if (score == (sc_mat[x - 1][y] + PEN_DEL))
		dir = 3;
	else if (score == (sc_mat[x][y - 1] + PEN_INS))
		dir = 4;
	else
		dir = 0;

	dir_mat[x][y] = dir;
	sc_mat[x][y] = score;

	return score;
}

__global__ void sw_GPU(char ** query, char ** reference, int *** sc_mat_list, char *** dir_mat_list, int * res, char ** simple_rev_cigar, max_supp_t * max_supp) {    
	// srand(time(NULL)); 
	int n = blockIdx.x;
	
	int ** sc_mat = sc_mat_list[n];
	char ** dir_mat = dir_mat_list[n];

	int score;
	int max, maxi, maxj;

	// max_supp_t max_tmp = max_supp[n];
    __shared__ max_supp_t max_tmp;  

	// initialize the scoring matrix and direction matrix to 0 
    if(threadIdx.x == 0) {
        max_tmp.v[D] = PEN_INS;
        max_tmp.v[X] = PEN_INS;
        max_tmp.v[Y] = PEN_INS;

        for (int i = 0; i < S_LEN + 1; i++) {
            for (int j = 0; j < S_LEN + 1; j++) {
                sc_mat[i][j] = 0;
                dir_mat[i][j] = 0;
            }
        }

        // Necessary initialization for vertical comparison of scores
        max_tmp.i[Y] = INT_MAX;
        max_tmp.j[Y] = INT_MAX;
    }

	// compute the alignment
	for (int i = 1; i < S_LEN; i++) {
		// Handle diagonal cells
		if(threadIdx.x == 0) {
            score = setDirScore(query[n], reference[n], sc_mat, dir_mat, i, i);
            if(score > max_tmp.v[D]) {
                max_tmp.v[D] = score; max_tmp.i[D] = i; max_tmp.j[D] = i;
            }
        }
		// Handle vertical/horizontal cells
        __syncthreads();
		for(int j = i+1; j < S_LEN; j++) {
            if(threadIdx.x == 0) {
                score = setDirScore(query[n], reference[n], sc_mat, dir_mat, i, j);
                if(score > max_tmp.v[X]) {
                    max_tmp.v[X] = score; max_tmp.i[X] = i; max_tmp.j[X] = j;
                }
            } else if(threadIdx.x == 1) {
                score = setDirScore(query[n], reference[n], sc_mat, dir_mat, j, i);
                if(score > max_tmp.v[Y] || (score == max_tmp.v[Y] && j<max_tmp.i[Y])) {
                    max_tmp.v[Y]= score; max_tmp.i[Y] = j; max_tmp.j[Y] = i;
                }
            }
		}
	}

    if(threadIdx.x == 0) {
        int eq_maxes_idx[3];
        int eq_maxes_count = 0;
        int min_ij_score = INT_MAX;
        int ij_score;
        max = PEN_INS;
        maxi = -1; maxj = -1;
        // Find maximum value
        for(int i=0; i<3; i++)
            if(max_tmp.v[i] > max)
                max = max_tmp.v[i];
        for(int i=0; i<3; i++)
            if(max_tmp.v[i] == max) {
                eq_maxes_idx[eq_maxes_count] = i;
                eq_maxes_count++;
            }
        for(int i=0; i<eq_maxes_count; i++) {
            ij_score = max_tmp.i[eq_maxes_idx[i]]*(S_LEN)+max_tmp.j[eq_maxes_idx[i]];
            if(ij_score < min_ij_score) {
                min_ij_score = ij_score;
                maxi = max_tmp.i[eq_maxes_idx[i]];
                maxj = max_tmp.j[eq_maxes_idx[i]];
            }
        }

        res[n] = sc_mat[maxi][maxj];
        backtrace(simple_rev_cigar[n], dir_mat, maxi, maxj, S_LEN * 2);
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
    int **d_sc_mat_ptrs;
	int ***d_sc_mat_list;
	int ***d_sc_mat_list_ptrs = (int ***)malloc(N * sizeof(int **));
    cudaMalloc((void***)&d_sc_mat_list, N * sizeof(int **));
	for(int j=0; j<N; j++) {
		d_sc_mat_ptrs = (int **)malloc((S_LEN+1) * sizeof(int *));    
		cudaMalloc((void***)&d_sc_mat, (S_LEN+1) * sizeof(int *));
		for(int i=0; i<(S_LEN+1); i++) {
			cudaMalloc((void**) &(d_sc_mat_ptrs[i]), (S_LEN+1)*sizeof(int));
		}
		cudaMemcpy (d_sc_mat, d_sc_mat_ptrs, (S_LEN+1)*sizeof(int *), cudaMemcpyHostToDevice);
		d_sc_mat_list_ptrs[j] = d_sc_mat;
	}
	cudaMemcpy (d_sc_mat_list, d_sc_mat_list_ptrs, N*sizeof(int **), cudaMemcpyHostToDevice);

	char **d_dir_mat;
    char **d_dir_mat_ptrs;
	char ***d_dir_mat_list;
	char ***d_dir_mat_list_ptrs = (char ***)malloc(N * sizeof(char **));
    cudaMalloc((void***)&d_dir_mat_list, N * sizeof(char **));
	for(int j=0; j<N; j++) {
		d_dir_mat_ptrs = (char **)malloc((S_LEN+1) * sizeof(char *));    
		cudaMalloc((void***)&d_dir_mat, (S_LEN+1) * sizeof(char *));
		for(int i=0; i<(S_LEN+1); i++) {
			cudaMalloc((void**) &(d_dir_mat_ptrs[i]), (S_LEN+1)*sizeof(char));
		}
		cudaMemcpy (d_dir_mat, d_dir_mat_ptrs, (S_LEN+1)*sizeof(char *), cudaMemcpyHostToDevice);
		d_dir_mat_list_ptrs[j] = d_dir_mat;
	}
	cudaMemcpy (d_dir_mat_list, d_dir_mat_list_ptrs, N*sizeof(char **), cudaMemcpyHostToDevice);

    int * d_res;
    cudaMalloc((void**) &d_res, N*sizeof(int));

    char **d_simple_rev_cigar;
    char **d_simple_rev_cigar_ptrs = (char **)malloc(N * sizeof(char *));    
    cudaMalloc((void***)&d_simple_rev_cigar, N * sizeof(char *));
    for(int i=0; i<N; i++) {
        cudaMalloc((void**) &(d_simple_rev_cigar_ptrs[i]), S_LEN * 2 * sizeof(char));
    }
    cudaMemcpy (d_simple_rev_cigar, d_simple_rev_cigar_ptrs, N*sizeof(char *), cudaMemcpyHostToDevice);

    max_supp_t * d_max_supp;
    cudaMalloc(&d_max_supp, N*sizeof(max_supp_t));


    // Blocks and threads schema for GPU execution
    dim3 blocksPerGrid(N, 1, 1);
    dim3 threadsPerBlock(2, 1, 1);

    // Execution on GPU started
    time_start = get_time();

    sw_GPU<<<blocksPerGrid, threadsPerBlock>>>(d_query, d_reference, d_sc_mat_list, d_dir_mat_list, d_res, d_simple_rev_cigar, d_max_supp);
    CHECK_KERNELCALL();
	// The following function adds A LOT of latency! Don't use it for benchmarking purposes
    // cudaDeviceSynchronize();

    // Execution on GPU ended
    time_stop = get_time();
    printf("GPU Execution time: %.10f\n", time_stop-time_start);

	// Transfer data to host
	// int * gpu_res = (int *) malloc(sizeof(int)*N);
	int gpu_res[N];
	char gpu_simple_rev_cigar[N][S_LEN*2];
	char * tmp_pointer;

	cudaMemcpy((void *) gpu_res, d_res, sizeof(int)*N, cudaMemcpyDeviceToHost);
	for(int i=0; i<N; i++) {
		tmp_pointer = d_simple_rev_cigar_ptrs[i];
		cudaMemcpy((void *) gpu_simple_rev_cigar[i], tmp_pointer, sizeof(char)*S_LEN*2, cudaMemcpyDeviceToHost);
	}

    // Execution on CPU started
    time_start = get_time();

	sw_CPU(h_query, h_reference, h_sc_mat, h_dir_mat, h_res, h_simple_rev_cigar);
    
	// Execution on CPU ended
    time_stop = get_time();
    printf("CPU Execution time: %.10f [!]\n", time_stop-time_start);

	// Check if results are consistent
	// "Results" validation
	int ok, i, j;
	ok = 1;
	for(i=0; (i<N && ok); i++) {
		ok = gpu_res[i] == h_res[i];
	}

	if(ok)
		printf("[OK]\t'results' is consistent\n");
	else
		printf("[ERR]\t'results' is inconsistent\n");

	ok = 1;
	for(i=0; (i<N && ok); i++) {
		for(j=0; (j<S_LEN*2 && ok); j++) {
			ok = gpu_simple_rev_cigar[i][j] == h_simple_rev_cigar[i][j];
		}
	}
	i--;
	if(ok)
		printf("[OK]\t'rev_cigar' is consistent\n");
	else{
		printf("[ERR]\t'rev_cigar' is inconsistent [err on: %d]\n", i);
		// printf("Comparison:\n");
		// for(j=0; j<(S_LEN*2); j++) {
		// 	printf("%d ", gpu_simple_rev_cigar[i][j]);
		// }
		// printf("\n");
		// for(j=0; j<(S_LEN*2); j++) {
		// 	printf("%d ", h_simple_rev_cigar[i][j]);
		// }
		// printf("\n");
	}


	// Deallocation of memory
	CHECK(cudaFree(d_query));
    CHECK(cudaFree(d_reference));
    CHECK(cudaFree(d_sc_mat_list));
    CHECK(cudaFree(d_dir_mat_list));
    CHECK(cudaFree(d_res));
    CHECK(cudaFree(d_simple_rev_cigar));

	free(h_query);
	free(h_reference);
	free(h_sc_mat);
	free(h_dir_mat);
	free(h_res);
	free(h_simple_rev_cigar);

    return 0;
}