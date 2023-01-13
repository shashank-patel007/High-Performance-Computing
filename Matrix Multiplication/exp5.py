""" 
	Shashank Patel
	BE Comps
	Batch: C
	UID: 2018130036
	Experiment 5

"""

import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI
import sys
import numpy as np


np.random.seed(133)

# helper functions
def block_low(rank,p,m):                 
	return int( (rank * m)/p)

def block_high(rank,p,m):                
	return block_low(rank+1,p,m)-1

def block_size(rank,p,m):                
	return block_low(rank+1,p,m) - block_low(rank,p,m) 

def block_owner(index,p,m):              
	return int(((p*(index+1))-1)/m)




def print_matrix(m,matrix):            

	for i in range(m):
		print('\t'.join(map(str,matrix[i*m : (i+1)*m])))




def read_row_striped_matrix(matrix_a,matrix_b,m,rank,p,comm):


	if(rank==0):                                      # only process no. 0 performs the distribution operation #

		sendbuf_a=np.zeros(m*m,dtype=np.int64)        # sendbuf stores the randomly generated input matrix in the form of a 1-D array #
		displacement_a=np.zeros(p)                    # displacement array stores the index from which the data elements should be distributed for the particular process #
		counts_a=np.zeros(p,dtype=np.int64)           # counts stores the no. of data elements for each process #
		s_a=0                                         # s is a helper variable to store the cumulative sum progressively for each process #


		sendbuf_b=np.zeros(m*m,dtype=np.int64)        
		displacement_b=np.zeros(p)                    
		counts_b=np.zeros(p,dtype=np.int64)           
		s_b=0                                         


		for i in range(p):                          # calculates the displacement and count for each process #  
			counts_a[i]=m*block_size(i,p,m)
			displacement_a[i]=s_a
			s_a+=counts_a[i]

			counts_b[i]=m*block_size(i,p,m)
			displacement_b[i]=s_b
			s_b+=counts_b[i]


		for i in range(m):                          # randomly generates the input matrix #
			for j in range(m):
				sendbuf_a[i*m+j]=np.random.randint(1,50)
				sendbuf_b[i*m+j]=np.random.randint(1,50)

		sendbuf_a=np.array(sendbuf_a)
		sendbuf_b=np.array(sendbuf_b)


		# print("\nMatrix A:\n")
		# print_matrix(m,sendbuf_a)

		# print("\nMatrix B:\n")
		# print_matrix(m,sendbuf_b)


	else:  											# all other process will receive from process no. 0 #     
	                                    
		counts_a=np.zeros(p,dtype=np.int64)
		counts_b=np.zeros(p,dtype=np.int64)
		displacement_a=None
		displacement_b=None
		sendbuf_a=None
		sendbuf_b=None

		
	comm.Scatterv([sendbuf_a,counts_a,displacement_a,MPI.INT64_T],matrix_a,root=0)    # process 0 will scatter parts of the input matrix to the processes based on the displacement and count values
	comm.Scatterv([sendbuf_b,counts_b,displacement_b,MPI.INT64_T],matrix_b,root=0)    # process 0 will scatter parts of the input matrix to the processes based on the displacement and count values




def matrix_multiplication(matrix_a,matrix_b,matrix_c,rank,p,m,comm):


	block_size_a=block_size(rank,p,m)
	current_block_size=np.array(block_size(rank,p,m),dtype=np.int64)
	current_block_low=np.array(block_low(rank,p,m),dtype=np.int64)
	previous_block_size=np.zeros(1,dtype=np.int64)
	previous_block_low=np.zeros(1,dtype=np.int64)


	for i in range(p):
		comm.Isend(current_block_size,dest=(rank+1)%p,tag=1)                          # Isend is asynchronous send call, the ith process sends its block size to the (i+1)th process
		req=comm.Irecv(previous_block_size,source=(rank-1+p)%p,tag=1)				  # Irecv is asynchronous receive call, the ith process receives previous block's size from (i-1)th process
		req.wait()																	  # wait to ensures the value is received


		comm.Isend(current_block_low,dest=(rank+1)%p,tag=1)							  # Isend is asynchronous send call, the ith process sends its low index to the (i+1)th process
		req=comm.Irecv(previous_block_low,source=(rank-1+p)%p,tag=1)				  # Irecv is asynchronous receive call, the ith process receives previous block lowe index from (i-1)th process 
		req.wait() 																	  # wait to ensures the value is received

		temp_b=np.zeros(m*previous_block_size,dtype=np.int64)						  # temp matrix to store the rows of matrix B received from the (i-1)th process
		# print(temp_b)

		comm.Isend([matrix_b,MPI.LONG],dest=(rank+1)%p,tag=0)						  # asynchronously send a process's local matrix to the next process
		# req=comm.Irecv([temp_b,MPI.INT64_T],source=(rank-1+p)%p,tag=0)
		# req.wait()
		comm.Recv([temp_b,MPI.LONG],source=(rank-1+p)%p)							  # receive matrix b from the previous process
		# print(temp_b)

		comm.barrier()
		
		current_block_low=previous_block_low[0]
		current_block_size=previous_block_size[0]
		matrix_b=temp_b


		for i in range(block_size_a):												  		  # matrix multiplication for loop
			for j in range(current_block_low,current_block_low+current_block_size):
				for k in range(m):
					matrix_c[i*m+k]+=matrix_a[i*m+j] * matrix_b[(j-current_block_low)*m+k]    # the result as it gets calculated is aggregated in matrix c




def print_row_striped_matrix(matrix_c,rank,p,m,comm):    

	displacement=np.zeros(p,dtype=np.int64)
	counts=np.zeros(p,dtype=np.int64)
	s=0

	for i in range(p):
		counts[i]=m*block_size(i,p,m)
		displacement[i]=s
		s+=counts[i]

	recvbuf=np.zeros(sum(counts),dtype=np.int64)							     # recvbuf (1-D array) stores the collected portions of the solution matrix from each process, its size is n^2 #

	comm.Gatherv(matrix_c,[recvbuf,counts,displacement,MPI.INT64_T],root=0)      # process 0 gathers the sub-parts from other processes using respective diasplacement and count values #

	# if rank==0:
	# 	print("\nSolution Matrix:\n")
	# 	print_matrix(m,recvbuf)


def main():

	MPI.Init()

	comm=MPI.COMM_WORLD
	rank = comm.Get_rank()                          # process id #
	p=comm.Get_size()


	if(len(sys.argv)<2):                            # size of the input matrix should be present in the command line arguments #
		print("Please provide the matrix dimensions M in the command line arguments")
		MPI.Finalize()
		exit()
	else:
		m=int(sys.argv[1])

	start_time=MPI.Wtime()

	matrix_a=np.zeros(m*block_size(rank,p,m),dtype=np.int64)     # initialising the input matrix A #

	matrix_b=np.zeros(m*block_size(rank,p,m),dtype=np.int64)     # initialising the input matrix B #

	matrix_c=np.zeros(m*block_size(rank,p,m),dtype=np.int64)

	read_row_striped_matrix(matrix_a,matrix_b,m,rank,p,comm) 

	matrix_multiplication(matrix_a,matrix_b,matrix_c,rank,p,m,comm)

	print_row_striped_matrix(matrix_c,rank,p,m,comm)

	end_time=MPI.Wtime()


	if rank==0:                                     # 0th process will store the total execution time a file for plotting graphs #


		print("P:{} N:{} Total execution time:{}".format(p,m,end_time-start_time))

		if(p==1):
			fout=open("out1.txt","a")
			fout.write(str(end_time-start_time)+" ")
			fout.close()

		elif(p==2):
			fout=open("out2.txt","a")
			fout.write(str(end_time-start_time)+" ")
			fout.close()

		elif(p==3):
			fout=open("out3.txt","a")
			fout.write(str(end_time-start_time)+" ")
			fout.close()

		elif(p==4):
			fout=open("out4.txt","a")
			fout.write(str(end_time-start_time)+" ")
			fout.close()

		else:
			fout=open("out5.txt","a")
			fout.write(str(end_time-start_time)+" ")
			fout.close()


	MPI.Finalize()

if __name__=="__main__":
	main()
