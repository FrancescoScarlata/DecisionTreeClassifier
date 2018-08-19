'''
At first i'm taking the script from the https://www.youtube.com/watch?v=LDRbO9a6XPU tutorial about the decision tree.
'''
from DecisionTree import DecisionTree
from CsvReader import CsvReader

import numpy as np
import math
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os.path

# ------ New Functions ----

def readDataset():
	'''
		This makes the input calls to have the paths to the dataset and the header.
		It will return the CvsReader Class with the dataset and the header.
	'''
	parentPath=str(Path(__file__).resolve().parents[1])

	filename_data=input("Please write the relative path of you dataset file (i.e. \"Car/data.csv\"): ")
	filename_data=os.path.join(parentPath,"Datasets",filename_data)

	filename_header=input("Please write the relative path of you dataset file (i.e. \"Car/header.csv\"): ")
	filename_header=os.path.join(parentPath,"Datasets",filename_header)

	numericIndex=input("Insert the indexes of the columns that are numerics, if any. Divide the index with a ,(comma): ")

	numericIndex=numericIndex.split(",")
	#print ("Empty: "+str(numericIndex))
	if(len(numericIndex)>0 and not numericIndex[0]==''):
		numericIndex=[int(i) for i in numericIndex]
		print (numericIndex)
		#print("len of num: "+str(len(numericIndex)))
	else:
		numericIndex=None

	dataset=CsvReader(filename_data,filename_header,numericIndex)
	return dataset


def divideTheDataset(dataset):
	'''
		This method will divide the dataset in 2: test set and training set.
		It will return the training set and the test set.
		This will divide the whole dataset in 75% training and 25% test set.
	'''
	training_set=list()
	test_set=list()
	for i in range (0, len(dataset)):
		if(i%4==0):
			test_set.append(dataset[i])
		else:
			training_set.append(dataset[i])
	
	return training_set, test_set
	
def createThesubsets(training, number_of_block):
	'''
		This will create the subsets dividing the training set.
		The training are the rows of the training set
		The number of block is the number of block in which divide the training set.
		If the number is higher of the number of cardinality of the training set, it will choose the len(training)
		if the number is lower of len(training) there are 2 case:
			a) the number divides in equal blocks
			b) the number doesn't divide in equal blocks. In this case we'll find a lower number that divides equally the set.
	'''
	if(number_of_block>len(training)):	#if it is higher. it becomes equal to the number of elements.
		number_of_block=len(training)
	else:
		value=len(training)%number_of_block
		print("starting number of block: "+str(number_of_block))
		print("module: "+str(value))
		
		while(not value==0):	#if the number doesn't divide equally, lower it.
			number_of_block-=value
			value=len(training)%number_of_block
			print("number of block: "+str(number_of_block))
			print("module: "+str(value))
		
		if(number_of_block==1):
			number_of_block=len(10)
	print("Creating the validation set with "+str(number_of_block)+" subsets")
	blocksize=int(len(training)/number_of_block)
	sets = [training[x:x+blocksize] for x in range(0, len(training), blocksize)]
	#for set in sets:
	#	print (set)
	return sets

def lossFunction(set, tree):
	'''
		This function will take the set of example, will classify them and if the label is wrong, it will add a +1 to the sum
		It will return the amount of errors
	'''
	loss=0
	for row in set:
		measure=tree.classify(row,tree.root)
		if(not measure==row[-1]):
			loss+=1
			#print("the prediction: "+str(measure)+" ; the correct label: "+ str(row[-1]))

	#print ("The loss is: "+str(loss))
	print ('.' , end='', flush=True)
	return loss

def external_cross_validation(sets, number_of_nodes, size_ts):
	'''
		This is method will determine the mean estimation of goodness of the parameter of nodes in this dataset.
		This will return the mean error on the validations subsets.
	'''
	print("\n[External CV]\nNumber of sets: "+str(len(sets))+", number of nodes: "+str(number_of_nodes)+", size of the training set: "+str(size_ts))
	errors=list()
	training_set=list()
	
	#loop to make len(sets) trees and determine trees and the losses of each validation set.
	for i in range(0, len(sets)):
		for j in range(0,len(sets)):
			if(not i==j):
				training_set+=sets[j]
		#create the tree with N-1 blocks
		tree=DecisionTree(training_data,header,number_of_nodes)
		#determine the error function on the 1 block call validation set
		errors.append(len(sets)/size_ts*lossFunction(sets[i],tree))
	
	error=np.mean(errors)
	print ("\n[External CV]\nThe Mean error is: "+ str(error)+" with the number of nodes: "+ str(number_of_nodes)+ " (real number of Nodes: "+str(tree.nodes)+")")
	return error
	
# --- Old Functions (except the iterative tree building) ---	

def internal_cross_validation(sets,starting_num_of_nodes, ending_num_of_nodes, different_nodes_values, size_ts):
	'''
		This method will search the best value of the number of nodes between the starting number of nodes and the ending number of nodes.
		The input are the subsets made, 2 effective number of nodes. 
		The starting is the left border of the interval, the ending is the right border.
		The different_nodes_values is the number of elements that the internal cross validation should confront. 
			1 means it will check just one values. +2 means different values
		It will return the correct number of nodes given those number of nodes.
	'''
	print("[INTERNAL CV]\nCalculating between "+str(starting_num_of_nodes)+" to "+str(ending_num_of_nodes)+"=(2^(d+2)) with "+str(different_nodes_values)+" different values in this interval")
	if(starting_num_of_nodes>ending_num_of_nodes):
		print("Error: starting should be lower of the ending one")
	
	if(different_nodes_values<=0):
		print("Error: the different_nodes_values should be >0 (maybe at least 2)")
	
	# new part
	interval= [[0,starting_num_of_nodes], [1,ending_num_of_nodes]]
	
	index_error_list=list()
	#the 2 limits, the starting and the ending
	index_error_list.append([starting_num_of_nodes,external_cross_validation(sets,starting_num_of_nodes,size_ts)])
	index_error_list.append([ending_num_of_nodes,external_cross_validation(sets,ending_num_of_nodes,size_ts)])
	
	currentValues=2
	
	meanNodeTimes=[starting_num_of_nodes,0]
	while(different_nodes_values>currentValues):
		meanNode=int((interval[0][1]+interval[1][1])/2)
		meanNError=external_cross_validation(sets,meanNode,size_ts)
		
		#checking the error with the left border of the interval
		if(meanNError<index_error_list[interval[0][0]][1]):
			index_error_list.insert(interval[0][0]+1, [meanNode,meanNError] )
			interval[0]=[interval[0][0]+1,meanNode]
			interval[1][0]=interval[1][0]+1
		
		#checking the error with the right border of the interval
		if(meanNError<=index_error_list[interval[1][0]][1]):
			index_error_list.insert(interval[1][0], [meanNode,meanNError] )
			interval[1][1]=meanNode
		#checking if the left and the right borders have the same value
		if(interval[0][1]==interval[1][1]):
			if (different_nodes_values>currentValues):
				interval[0][0]=interval[0][0]-1
				interval[0][1]=index_error_list[interval[0][0]][0]
		if(not meanNode==meanNodeTimes[0]):
			meanNodeTimes=[meanNode,0]
		else:
			meanNodeTimes[1]+=1
			if(meanNodeTimes[1]==1):
				break
		
		
		for row in interval:
			print(row)
		currentValues+=1
	
	print ("[INTERNAL CV] To find the chosen element, it took "+str(currentValues)+" steps.")
	return interval[1][1]
	
	'''
	nodes_to_test=np.linspace(starting_num_of_nodes, ending_num_of_nodes, different_nodes_values)
	
	index_error_list=list()
	
	for i in range(0, len(nodes_to_test)):
		index_error_list.append([int(nodes_to_test[i]),external_cross_validation(sets,int(nodes_to_test[i]),size_ts)])
	
	# all the errors are calculated. we need to find the argmin
	min_index=0
	for i in range(0,len(index_error_list)):
		if(index_error_list[min_index][1]>index_error_list[i][1]):
			min_index=i
	
	print("[INTERNAL CV result] The number of nodes should be: "+str(index_error_list[min_index][0]))
	return index_error_list[min_index][0]	
	'''
	
def getDatasetInfo(dataset, training_data):
	# - - - Informations - - - 
	print("\n[DATASET INFOS]:\n")
	print("size of the dataset: "+ str(len(dataset.data)))
	print("size of the training set (m): "+ str(int(len(training_data))))
	print("number of attributes (d)= "+str(len(dataset.data[0])-1))
	print("\nNodes(N) <<(3^(d))= "+str(int(	3**(len(dataset.data[0])-1)	))+"\n")
	number_of_nodes=int(2**(len(dataset.data[0])-2))
	try:
		bignumber=int((2*(len(dataset.data[0])-2)*math.e)**number_of_nodes)
	except:
		print("possible statistical classificators cardinality is too big to be calculated!")
		bignumber=np.nan
	print("possibile statistical classificators <=(2de)^N = "+str(bignumber)+" where N= 2^(d-1)\n")
	
	
def graphicWithDifferentParameters(training_subsets,training_set,size_ts,test_set, header,starting_num_of_nodes,ending_num_of_nodes,different_nodes_values ):
	'''
		This method will not do internal cv.
		but just build the trees with different parameters, determine the validation error via external_cross_validation and then use the tree with those nodes on the test set.
		This will show an image with the test error and validation error for each number of nodes.
	'''
	nodes=list()
	valerr=list()
	testerr=list()
	startTime=time.clock()
	#Let's choose the different values linearly
	nodes_to_test=np.linspace(starting_num_of_nodes, ending_num_of_nodes, different_nodes_values)

	
	#Creates the list with: number of nodes; validation error; test error
	for i in range(0,len(nodes_to_test)):
		tree=DecisionTree(training_set,header,int(nodes_to_test[i]))
		nodes.append(int(nodes_to_test[i]))
		valerr.append(external_cross_validation(training_subsets,int(nodes_to_test[i]), size_ts))
		testerr.append((1/len(test_set))*lossFunction(test_set,tree))
	endTime= time.clock()
	print("\ntime to calculate the val & test err is: "+str(int(endTime-startTime))+" secs")
	
	# figure 1: nodes, val err and test err  calculated
	plt.figure(1)
	plt.clf()
	plt.title('Graphic with number of nodes, validation error and test error')

	plt.plot(nodes, valerr, label= 'validation error')	
	plt.plot(nodes, testerr, label= 'test error')
	#max number of nodes. This classifier, when will stop an a determined number of nodes even if the number given are more
	plt.axvline(x=tree.nodes, color='green', label= 'max Number of Nodes')
	
	plt.grid(True)
	plt.xlabel('nodes')
	plt.ylabel('percentage')
	plt.legend(loc='best')
	plt.show()

	
def learnWithCrossVal(sets, header, starting_non, ending_non, different_nv, size_ts):
	'''
	This method is used to learn the best number of nodes in the interval choosing within 10 elements in this interval.
	'''
	print("\n[learn With CV]")
	startTime=time.clock()
	training_set=sets[0]
	for i in range(1,len(sets)):
		training_set+=sets[i]
	i_star=internal_cross_validation(sets, starting_non, ending_non, different_nv, size_ts)
	#creates the tree with the values found by the internal cv

	tree=DecisionTree(training_set,header,i_star)
	
	#print("The following is the resulting tree:\n")
	#print(tree)
	
	print("\nThe test error  with the nodes of the tree chosen by the internal cv ("+str(i_star)+")  is: "+str(lossFunction(test_data,tree))+ "\n(this is the number of errors that this predictor does on the test set)")
	#print("Even if the number chosen is " +str(i_star)+", the real number of nodes in the decision tree is: "+str(tree.nodes))
	endTime= time.clock()
	print("\ntime to calculate the CV is: "+str(int(endTime-startTime))+" secs")
	
if __name__=='__main__':
	dataset=readDataset()			
	training_data,test_data=divideTheDataset(dataset.data)
	header=dataset.header
	size_ts=len(training_data)
	#the teacher says that this is a good number.
	number_of_classificators=int(10)
	d= len(training_data[0])-1
	
	getDatasetInfo(dataset,training_data)
	
	# - - - Calculations - - -
	print("[CALCULATION]\n")
	
	# creates the subsets for the c.v.
	sets=createThesubsets(training_data,number_of_classificators)
	#finds the best values of the different values
	print("size ts: "+str(size_ts))
	
	learnWithCrossVal(sets, header, 3, 2**(d+2), 15, size_ts)

	graphicWithDifferentParameters(sets,training_data, size_ts, test_data,header, 3, 2**(d+2), 20)

	