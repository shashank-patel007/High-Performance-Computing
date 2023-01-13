""" 
	Shashank Patel
	Batch: C
	UID: 2018130036
	Experiment 7
"""

from mpi4py import MPI
import numpy as np
import math

a = 1                   # Length of string
c = 2                   # Constant for string vibration equation
m = 0                   # Number of time steps
n = 0                   # Number of space steps
T = 1                   # Period of vibration


def block_low(id,p,n):
	return int( (id * n)/p)

def block_high(id,p,n):
	return block_low(id+1,p,n)-1

def block_size(id,p,n):
	return block_low(id+1,p,n) - block_low(id,p,n) 

def block_owner(index,p,n):
	return int(((p*(index+1))-1)/n)

def calculate_displacement(id, p, split, L, comm):
    temp1 = temp2 = 0                       
    for j in range(1, len(split) - 1):      # Filling m rows
        arr = split[j]                      # Reference to jth row of data
        req = comm.isend(split[j][0], dest = (id - 1 + p)%p, tag = 1)               # Sending extreme left point to left neighbouring process
        req = comm.isend(split[j][len(split[0]) - 1], dest = (id + 1)%p, tag = 2)   # Sending extreme right point to right neighbouring process
        req = comm.irecv(source = (id - 1 + p)%p, tag = 2)                          # Receiving left ghost point from left neighbouring process
        temp1 = req.wait()
        req = comm.irecv(source = (id + 1)%p, tag = 1)                              # Receiving right ghost point from right neighbouring process
        temp2 = req.wait()
        arr = np.append(temp1, arr)         # Appending left ghost point to jth row
        arr = np.append(arr, temp2)         # Appending right ghost point to jth row
        for i in range(len(split[0])):      # Calculating displacement for (j+1)th row
            if not ((id == 0 and i == 0) or (id == (p - 1) and i == len(split[0]) - 1)):    # Skipping calculations for 0th column of 0th processes and last column of (p-1)th process
                split[j + 1][i] = (2*(1 - L) * arr[i + 1]) + (L * (arr[i + 1 + 1] + arr[i - 1 + 1])) - (split[j - 1][i])    # Applying formula
    split = comm.gather(split, root = 0)    # Gathering results from all processes
    if id == 0:                             # Collecting results inside a new array "solution"
        solution = split[0]
        for i in range(1, len(split)):      # Stacking columns against each other in one array
            solution = np.hstack((solution, split[i]))
        print("Solution:\n", np.round(solution, 2)) # Printing solution

def main():
    comm = MPI.COMM_WORLD           
    id = comm.Get_rank()            
    p = comm.Get_size()             
    h = k = L = 0                   
    u = None                        
    split = []                      # Split array for scattering columns of u
    start_time = 0

    if id == 0:
        m, n = tuple(map(int, input("Enter the values of m and n: ").split()))      # Getting m and n values from user
        start_time = MPI.Wtime()
        h = a/n                     
        k = T/m                     
        L = np.power((k * c / h),2)

        u = np.zeros((m + 1, n + 1))    

        for i in range(1, len(u[0]) -1):    
            u[0][i] = math.sin(math.pi * i*h)
        
        for i in range(1, len(u[0]) - 1):   
            u[1][i] = (L/2) * (u[0][i + 1] + u[0][i - 1]) + (1 - L) * (u[0][i]) + (k * 0)

        split = np.array_split(u, p, 1)     

    h = comm.bcast(h, root = 0)             
    k = comm.bcast(k, root = 0)             
    L = comm.bcast(L, root = 0)             

    split = comm.scatter(split, root = 0)   # Scatter array to processes
    calculate_displacement(id, p, split, L, comm)   
    end_time = MPI.Wtime()
    if id==0:
        print("\nExecution Time:", end_time - start_time)

if __name__=="__main__":                    
	main()