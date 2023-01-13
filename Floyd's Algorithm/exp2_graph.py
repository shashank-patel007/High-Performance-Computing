import matplotlib.pyplot as plt
import numpy as np
def values(fname):
    file = open(fname)
    l = []
    for line in file.readlines():
        l.append(float(line.split()[0]))
    return l
l1 = values("process1.txt")
l2 = values("process2.txt")
l3 = values("process3.txt")
l4 = values("process4.txt")
matrix_sizes = [10,50,100,500,1000] 
plt.plot(matrix_sizes,l1,label="Process 1")
plt.plot(matrix_sizes,l2,label="Process 2")
plt.plot(matrix_sizes,l3,label="Process 3")
plt.plot(matrix_sizes,l4,label="Process 4")
plt.legend()
plt.xlabel("Size of matrix")
plt.ylabel("Time in seconds")
plt.show()