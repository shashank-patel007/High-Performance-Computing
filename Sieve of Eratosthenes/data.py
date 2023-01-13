import matplotlib.pyplot as plt
import numpy as np
file1 = open("output1.txt")
l1 = []
file2 = open("output.txt")
l2 = []
for line in file1.readlines():
    l1.append(float(line.split()[1]))
for line in file2.readlines():
    l2.append(float(line.split()[1]))
plt.plot(l1)
plt.plot(l2)
plt.xlabel("Powers of 10")
plt.ylabel("Time in seconds")
plt.show()