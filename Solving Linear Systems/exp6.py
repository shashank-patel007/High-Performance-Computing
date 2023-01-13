""" 
	Shashank Patel
	Batch: C
	UID: 2018130036
	Experiment 6

"""

import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI
import sys
import numpy as np


def block_low(rank,p,m):                 # function to calculate the lowest index of the range for the given process p #
	return int( (rank * m)/p)

def block_high(rank,p,m):                # function to calculate the highest index of the range for the given process p #
	return block_low(rank+1,p,m)-1

def block_size(rank,p,m):                # function to calculate the total block size(range) for the process p #
	return block_low(rank+1,p,m) - block_low(rank,p,m) 

def block_owner(index,p,m):              # function to calculate the owner process of the data element - index #
	return int(((p*(index+1))-1)/m)



def row_wise_distribution(m,rank,p,comm):

	# f=open("input.txt","r")
	split=[]							# stores the slices of the input array associated with each process
	arr=[]								# initialise a 2d array for for storing the randomly generated input

	# for i in range(m):
	# 	line=f.readline().split()
	# 	temp=[float(l) for l in line]
	# 	arr.append(temp)

	if rank==0:							# process = 0 generates a random 2d array of coefficients

		arr=np.random.randint(1,10,size=(m,(m+1)))
		# print(arr)

		for i in range(p):              # process 0 divides the array created above among the processes based on their block size 

			low=block_low(i,p,m)		# block low index
			high=block_high(i,p,m)		# block high index
			a=[]						# temporary array to store the block of rows associated with each process

			for j in range(low,high+1):
				a.append(arr[j])		# append the respective rows belonging to a process to the temporary array

			split.append(a)				# append the slice(block) associated with each row to the slice array	

	matrix=comm.scatter(split,root=0)   # process 0 scatters the input array of coeeficients among the proceses based on the slice array

	return matrix						# matrix stores the block of rows associated with each process





def gaussian_elimination(comm,matrix,rank,p,m):

	marked=[]							# an array to store which row has already been used as a pivot 
	low=block_low(rank,p,m)				# block low index

	for pivot in range(m):				# iterate through all the rows of the original coefficient matrix

		max_pivot=[0,0]  				# max pivot is a list of type [value,index]


		for i in range(len(matrix)):	# iterate through the rows of the local partition matrix

			if i+low not in marked:		# if the index of the current row has not been used as a pivot before

				if abs(matrix[i][pivot] > max_pivot[0]):  # if the absolute value of the element in the corresponding row is greater than the current pivot element
					max_pivot[0]=abs(matrix[i][pivot])	  # update the pivot value and index
					max_pivot[1]=i + low


		max_pivot_element=comm.allreduce(max_pivot,op=MPI.MAXLOC)  # reduce operation to find the maximum value of the pivot across all the process
		marked.append(max_pivot_element[1])						   # append the pivot row's index to the marked array

		pivot_owner=block_owner(max_pivot_element[1],p,m)		   # pid of the owner of the pivot row
		marked=comm.bcast(marked,root=pivot_owner)			       # broadcast thr pivot row's index to all processes


		if rank==pivot_owner:		    # if the current process is the pivot row's owner
			subtract_row=comm.bcast(matrix[max_pivot_element[1] - block_low(pivot_owner, p, m)],root=pivot_owner)     # current process broadcasts the pivot row to all thr process
		else:
			subtract_row=comm.bcast(None,pivot_owner)			   # all other processes will receive from the owner of the pivot row

		temp=[]

		for i in range(len(matrix)):    # iterate over the rows of the local matrix (block)

			if (i+low) not in marked:	# only consider the unmarked rows for the update operation

				if list(matrix[i]) != list(subtract_row):		   # if the ith row is not the pivot row

					a=np.subtract(matrix[i], np.multiply(subtract_row, matrix[i][pivot]/subtract_row[pivot]))		 # update the row according to the gaussian elimination rule
					temp.append(list(np.round_(a,2)))		       # append the updated row to a temporary array

				else:
					a=subtract_row								   # if the ith row is thr pivot, it does not participate in the upadte operation and is appended as it is in the temporary array
					temp.append(subtract_row)

			else:						# all marked rows are appended as it is, since they have already been used before
				temp.append(matrix[i])

		matrix=temp 					

	return matrix,marked



def back_substitution(comm,matrix,rank,p,m,marked,solution):

	for i in reversed(range(len(marked))):      # iterate over the rows of the original input matrix in the reversed order i.e m-1 to 0 

		pid=block_owner(marked[i],p,m)			# pid of the owner of the ith row

		if rank==pid:							# if the current process is the owner of the ith row

			solution_row=matrix[marked[i] - block_low(rank,p,m)]	 
			ans=round(solution_row[-1]/solution_row[i],2)    # value of the coefficient of xi 

			solution[i]=ans 					

			ans=comm.bcast(ans,root=pid)		

			for j in matrix:					

				if list(j)!=list(solution_row):	
					x=j[i]
					j[-1] -= x*ans
					j[i]=0

		else:
			ans=comm.bcast(None,root=pid)       # if the current process is not the owner of the ith row, it will receive xi from the owner and simply update its rows

			for j in matrix:					
				x=j[i]
				j[-1]-=x*ans
				j[i]=0

		solution=comm.bcast(solution,pid)	    

	return solution


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

	solution=[0]*m
	matrix = row_wise_distribution(m,rank,p,comm) 

	matrix,marked=gaussian_elimination(comm,matrix,rank,p,m)

	solution=back_substitution(comm,matrix,rank,p,m,marked,solution)

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

	MPI.Finalize()

if __name__=="__main__":
	main()