/* 
 * Shashank Patel
 * Batch C
 * 2018130036
 */

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char ** argv) {
  int i;
  int n;
  int index;
  int size;
  int prime;
  int count;
  int global_count;
  int first;
  long int high_value;
  long int low_value;
  int comm_rank;
  int comm_size;
  char * marked;
  double runtime;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  
  MPI_Barrier(MPI_COMM_WORLD);
  runtime = -MPI_Wtime();
  
  // Check for the command line argument.
  if (argc != 2) {
    if (comm_rank == 0) printf("Please supply a range.\n");
    MPI_Finalize();
    exit(1);
  }
  
  n = atoi(argv[1]);
  
  // Exit if all the primes used for sieving are not all held by root process.
  if ((2 + (n - 1 / comm_size)) < (int) sqrt((double) n)) {
    if (comm_rank == 0) printf("Too many processes.\n");
    MPI_Finalize();
    exit(1);
  }
  
  // Bifurcation of array to processes
  low_value  = 2 + (long int)(comm_rank) * (long int)(n - 1) / (long int)comm_size;
  high_value = 1 + (long int)(comm_rank + 1) * (long int)(n - 1) / (long int)comm_size;
  size = high_value - low_value + 1;
  
  marked = (char *) calloc(size, sizeof(char));
  
  if (marked == NULL) {
   printf("Cannot allocate enough memory.\n");
   MPI_Finalize();
   exit(1);
  }
  
  if (comm_rank == 0) index = 0;
  prime = 2;
  
  do { //checking if the index of assigned block is the starting point
    if (prime * prime > low_value) {
      first = prime * prime - low_value;
    } else {
      if ((low_value % prime) == 0) first = 0;
      else first = prime - (low_value % prime);
    }
    // marking the multiples
    for (i = first; i < size; i += prime) marked[i] = 1;
    
    if (comm_rank == 0) {
      while (marked[++index]);
      prime = index + 2;
    }
    // broadcasting the prime number for other processes
    if (comm_size > 1) MPI_Bcast(&prime,  1, MPI_INT, 0, MPI_COMM_WORLD);
  } while (prime * prime <= n);
  
  count = 0;
  
  for (i = 0; i < size; i++) if (marked[i] == 0) count++;
  // reducing all the matrices to sum all the prime numbers
  if (comm_size > 1) {
    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    global_count = count;
  }
  
  runtime += MPI_Wtime();
  
  if (comm_rank == 0) {
    printf("In %f seconds we found %d primes less than or equal to %d.\n",
		runtime, global_count, n);
  }
  
  MPI_Finalize();
  return 0;
}


