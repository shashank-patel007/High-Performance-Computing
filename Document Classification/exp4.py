""" 
	Shashank Patel
	BE Comps
	Batch: C
	UID: 2018130036
	Experiment 4
"""

import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI
import sys
import numpy as np
from collections import defaultdict

file_name_msg=1										# file name message tag
vector_msg=2										# vector message tag
empty_msg=3											# empty message tag


def profiling(words,name,vector): 				# function to create the profile vector of the given document

	f=open(name,"r")								# open the document in the read mode
	text=f.read().split()

	for word in text:								# for each word in the document 
		if word in words:							# if the word is in the dictionary of words
			vector[word]+=1							# increment the count by 1




def worker(comm,rank,p,status,dict_file,words):

	
	comm.send(None,dest=0,tag=empty_msg)					# send empty message tag to the manager to inform that the worker process is active


	if rank==1:												# worker=0, i.e pid=1 readstthe dictionary of words from the dictionary file
		f=open(dict_file,"r")
		words=f.read().split()

	words=comm.bcast(words,root=1)							# worker=0, i.e. pid=1 broadcasts the dictionary of words to all other processes


	while True:												# while loop continues till the termination message is received from manager

		name=comm.recv(source=0, tag=file_name_msg, status=status)    # wokers receives a message from the manager 

		if name==None:							    		# if the message is empty it implies worker termination
			break									
		else:
			vector=defaultdict(int)							# profile vector initialisation
			profiling(words,name,vector)					# call the make profile function by providing the document file's name and the dictionary
			comm.send(vector,dest=0,tag=vector_msg)			# worker sends the profile vector to the manager
	




def manager(comm,rank,p,status,words):

	file_names=['text1.txt','text2.txt','text3.txt']		# names of the text documents to classify
	
	matrix=[[]]*len(file_names)								# matrix to store profile vectors of the documents
	
	terminated=0											# initialise count of terminated worker processes to 0
	assigned_count=0										# initialise count of assigned worker processes to 0
	assigned=np.zeros(p,dtype=np.int64)						# array to store the pid to which a document is assigned


	while terminated < p-1:									# while loop continues till the terminated equals p-1 since pid=0 is the manager

		buff=comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)      # manager receives a message from the worker
		src = status.Get_source()							# pid of the source worker process
		tag = status.Get_tag()								# tag associated with message sent by the woker


		if tag==vector_msg:									# if tag indicates that the received message contains the document profile vector
			matrix[assigned[src]]=buff						# store the profile vector at the corresponding file index


		if assigned_count < len(file_names):				# if there are files left for classification

			comm.send(file_names[assigned_count],dest=src,tag=file_name_msg)	   # send the file name to the active worker process
			assigned[src]=assigned_count					# assign the file's index to the cuurently active worker process
			assigned_count+=1								# increment the assigned count by 1

		else:	
			comm.send(None,dest=src,tag=file_name_msg)		# after all the documents are processed manager sends terminate message to worker
			terminated+=1									# increment the count of terminated workers by 1


	fout=open("output.txt","w")								# write 2d matrix into output file
	for i,vector in enumerate(matrix):
		fout.write("Document No. {} - ".format(i+1) + str(dict(vector))	+ "\n")	
	fout.close()											



def main():

	MPI.Init()												

	comm=MPI.COMM_WORLD										# communicator object
	rank = comm.Get_rank()									# process id
	status=MPI.Status()                          			# status object associated with the communicator 
	p=comm.Get_size()										# total number of processes 

	words=None

	if(len(sys.argv)<2):                           			# dictionary file should be given in the command line arguments #
		print("Provide dictionary name in CLI")
		MPI.Finalize()
		exit()
	else:
		dict_file=sys.argv[1]

	if(rank==0):											# manager process
		manager(comm,rank,p,status,words)
	else:
		worker(comm,rank,p,status,dict_file,words)          # worker process

	MPI.Finalize()

if __name__=="__main__":
	main()
