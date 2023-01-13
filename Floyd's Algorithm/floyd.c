/*
Experiment No. 2
Name: Shashank Patel
UID: 2018130036
Batch: C
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define INF 10000
#define BLOCK_LOW(id, p, n) ((id)*(n)/(p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW(id+1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_LOW(id+1, p, n) - BLOCK_LOW(id, p, n))
#define BLOCK_OWNER(index, p, n) ((((p)*((index)+1))-1)/(n))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void read_matrix(int matrix[], int n, int id, int p);
void gather_results(int matrix[], int n, int id, int p);
void compute_shortest_paths(int matrix[], int n, int id, int p);

int main(int argc, char* argv[]) 
{
	double time;
	int p, rank, n;
	int* matrix;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// If root process read the size of matrix
	if(rank==0)
	{
		scanf("%d",&n);
		time = -MPI_Wtime();
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	
	// Create matrix and read from file
	matrix = malloc(n*BLOCK_SIZE(rank, p, n) * sizeof(int));
	read_matrix(matrix, n, rank, p);
	
	compute_shortest_paths(matrix, n, rank, p);
	gather_results(matrix, n, rank, p);
	free(matrix);
	// print the elapsed time
	if(rank==0){
		time += MPI_Wtime();
		printf("%f\n", time);
	}

	
	MPI_Finalize();
	return 0;
}  



void read_matrix(int matrix[], int n, int id, int p)
{ 
	// Reads the matrix from a file
	int value;
	int* mat = NULL;
	int sum = 0;

	int *counts = malloc(sizeof(int)*p);
   int *displacements = malloc(sizeof(int)*p);

	for(int i = 0; i < p; i++) 
	{
		counts[i] = n*BLOCK_SIZE(i, p, n);
		displacements[i] = sum;
		sum += counts[i];
	}

	// If process 0 then read from file and scatter to other processes
	if (!id) {
		mat = malloc(n*n*sizeof(int));
		int num;
		for(int i = 0; i < n; i++)	{
			for(int j = 0; j < n; j++) {
				if(i == j)
				{
					mat[i*n+j] = 0;
				}
				else
				{
					mat[i*n+j] = rand()%10+1;
				}
			}
		}
		MPI_Scatterv(mat, counts, displacements, MPI_INT, matrix, counts[id], MPI_INT, 0, MPI_COMM_WORLD);

		// printf("Input Matrix:\n");
		// for(int i=0;i<n;i++){
		// 	for(int j=0;j<n;j++){
		// 		printf("%d ",mat[i*n+j]);
		// 	}
		// 	printf("\n");
		// }
		free(mat);
	} 
	// If not process 0 then receive from process 0
	else 
	{
		MPI_Scatterv(NULL, NULL, NULL, NULL, matrix, counts[id], MPI_INT, 0, MPI_COMM_WORLD);
	}
}

void gather_results(int matrix[], int n, int id, int p) 
{
	// Gathering all the local matrices into a common matrix and printing the solution
	int* mat = NULL;

	int rem_rows = n%p;
	int sum = 0;

	int *counts = malloc(sizeof(int)*p);
   int *displacements = malloc(sizeof(int)*p);

	for (int i = 0; i < p; i++) 
	{
		counts[i] = n*BLOCK_SIZE(i, p, n);
		displacements[i] = sum;
		sum += counts[i];
	}

	// If process 0 receive from all processes and print the result
	if (id==0) {
		mat = malloc(n*n*sizeof(int));
		MPI_Gatherv(matrix, counts[id], MPI_INT, mat, counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);
		
		// printf("Shortest paths:\n");
		// for(int i=0;i<n;i++){
		// 	for(int j=0;j<n;j++){
		// 		printf("%d ",mat[i*n+j]);
		// 	}
		// 	printf("\n");
		// }
		free(mat);
	} 
	// If not process 0 then send to process 0
	else 
	{
		MPI_Gatherv(matrix, counts[id], MPI_INT, mat, counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);
	}
}

void compute_shortest_paths(int matrix[], int n, int id, int p) 
{
	int* tmp = malloc(n*sizeof(int));
	for(int k = 0; k < n; k++)
	{
		// Get the process to which row k belongs
		int root = BLOCK_OWNER(k, p, n);
		if(id == root)
		{
			// Offset gets to the correct row within the local matrix
			int offset = k - BLOCK_LOW(id, p, n);
			for (int j = 0; j < n; j++)
				tmp[j] = matrix[offset * n + j];
		}
		MPI_Bcast(tmp, n, MPI_INT, root, MPI_COMM_WORLD);

		for(int i = 0; i < BLOCK_SIZE(id, p, n); i++)
			for(int j = 0; j < n; j++) 
				matrix[i*n+j] = MIN(matrix[i*n+j], matrix[i*n+k] + tmp[j]);
	}
	free(tmp);
}   