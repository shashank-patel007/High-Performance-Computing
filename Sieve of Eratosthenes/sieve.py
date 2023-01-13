import numpy as np
import time
t = -time.time()

n=1000000000
primes = np.zeros((n,1))

for i in range(n):
    primes[i]= 1
primes[0]=0
primes[1]=0
for i in range(2,n):
    if primes[i]==1:
        for j in range(2,n//i):
            primes[i*j]=0
t+=time.time()
print(n,t)