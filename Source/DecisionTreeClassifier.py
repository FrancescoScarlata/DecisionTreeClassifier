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
import argparse

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
	return index_error_list[interval[1][0]][1],interval[1][1]
	
def getDatasetInfo(dataset, training_data):
	# - - - Informations - - - 
	print("\n[DATASET INFOS]:\n")
	print("size of the dataset: "+ str(len(dataset.data)))
	print("size of the training set (m): "+ str(int(len(training_data))))
	print("number of attributes (d)= "+str(len(dataset.data[0])-1))
	print("\nNodes(N) <<(2^(d))= "+str(int(	2**(len(dataset.data[0])-1)	))+"\n")
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
	This method is used to learn the best tree parameter using different_nv steps.
	Given Argument:
		- sets = the sets used for the cross validation (sets are already separated, so this is a list of sets)
		- header = the header of the dataset
		- starting_non = the starting number of nodes (usually 3 or 1)
		- ending_non = the last element number of node to check in the interval
		- different_nv = how many steps this method is allowed to iterate
		- size_ts= the size of the overall training set
	Returns:
		- the tree that minimizes the cross validation error
	'''
	print("\n[learn With CV]")
	startTime=time.clock()
	training_set=sets[0]
	for i in range(1,len(sets)):
		training_set+=sets[i]
	accuracy,i_star=internal_cross_validation(sets, starting_non, ending_non, different_nv, size_ts)
	#creates the tree with the values found by the internal cv

	tree=DecisionTree(training_set,header,i_star)
	#print("The following is the resulting tree:\n")
	#print(tree)
	print("\nThe erCVi* is: "+str(accuracy))
	print("The test error  with the nodes of the tree chosen by the internal cv ("+str(i_star)+")  is: "+str(lossFunction(test_data,tree))+ "\n(this is the number of errors that this predictor does on the test set)")
	#print("Even if the number chosen is " +str(i_star)+", the real number of nodes in the decision tree is: "+str(tree.nodes))
	endTime= time.clock()
	print("\ntime to calculate the CV is: "+str(int(endTime-startTime))+" secs")
	return tree


def riskAnalysisValue(n_o_n, training_data,size_ts,header,isOn, deltaValue):
		'''
		This method will do the calculation for risk analysis approach:
		Given argument:
			- n_o_n= number of nodes
			- training_data = the training set
			- header= the header of the training set
			- isOn= boolean to say if should be used a isOn fuction or a Onlogn
		Return:
			The value of the estimation
		'''
		tree=DecisionTree(training_data,header,n_o_n)
		#check the training error
		tr_error= (lossFunction(training_data,tree)/size_ts)
		
		#calculating the majorant elements
		if(isOn):
			oh=(n_o_n+1)			 #this is O(n)
		else:
			oh=(n_o_n+1)*math.ceil(math.log2(d+3))+2*math.floor(math.log2(n_o_n))+1		#this is O(n*log(d))
		
		#calculating the value for the comparison
		value=tr_error + math.sqrt((1/(2*size_ts))*(oh+math.log(2/deltaValue)))
		
		return value

	
def learnWithTreeRisk(deltaValue, size_ts, training_data,header, numb_of_attr, isOn, different_nodes_values):
	'''
		This will use the statistical risk to find the best h.
		The inputs are:
			- the delta value e (0,1]
			- the size of the training set
			- the number of attributes of this training set
			- how many steps is allowed to take to converge
		It will determine the best number of nodes and return it.
		This result will be valid with at least probability 1-deltaValue
		
		the preference is given with the following function:
		w(h)= 2^(-|0(h)|) where |0(h)|= (Nh + 1)* ceiling(log2(d + 3))+floor(2log2(Nh) + 1= O(nh*log(d))
	'''
	print("\n[learn With tree Risk]")
	startTime=time.clock()
	results=list()
	
	# this interval is fixed. We should consider just 2^d, but because the attributes are not binary i prefer to not me ^d but d+1 or +2
	interval= [[0,3], [1,2**(numb_of_attr+2)]]
	
	results=list()
	#the 2 limits, the starting and the ending
	results.append([3,riskAnalysisValue(1, training_data,size_ts,header,isOn, deltaValue)])
	results.append([2**(numb_of_attr+2),riskAnalysisValue(2**(numb_of_attr+2), training_data,size_ts,header,isOn, deltaValue)])
	
	currentValues=2
	
	meanNodeTimes=[3,0]
	while(different_nodes_values>currentValues):
		meanNode=int((interval[0][1]+interval[1][1])/2)
		meanNError=riskAnalysisValue(meanNode, training_data,size_ts,header,isOn, deltaValue)
		
		#checking the error with the left border of the interval
		if(meanNError<results[interval[0][0]][1]):
			results.insert(interval[0][0]+1, [meanNode,meanNError] )
			interval[0]=[interval[0][0]+1,meanNode]
			interval[1][0]=interval[1][0]+1
		
		#checking the error with the right border of the interval
		if(meanNError<=results[interval[1][0]][1]):
			results.insert(interval[1][0], [meanNode,meanNError] )
			interval[1][1]=meanNode
		#checking if the left and the right borders have the same value
		if(interval[0][1]==interval[1][1]):
			if (different_nodes_values>currentValues):
				interval[0][0]=interval[0][0]-1
				interval[0][1]=results[interval[0][0]][0]
		if(not meanNode==meanNodeTimes[0]):
			meanNodeTimes=[meanNode,0]
		else:
			meanNodeTimes[1]+=1
			if(meanNodeTimes[1]==1):
				break
		currentValues+=1
		print("\nStep "+str(currentValues))
		for row in interval:
			print(row)
			print("results:"+str(results[row[0]][1]))
		
	print()
	print("The result with probability at least "+ str(1 -deltaValue)+" is: "+str(results[interval[1][0] ] [0])+".")
	print("The statistical error er(h) is less than "+str(results[interval[1][0] ] [1] ))
	endTime= time.clock()
	
	# Just debugging
	#for row in results:
	#	print(row)
		
	print("\ntime to calculate the tree risk is: "+str(int(endTime-startTime))+" secs")
	return results[interval[1][0] ][0]

	
if __name__=='__main__':

	ap = argparse.ArgumentParser(description="Additial feature, like debug and show of the tree")

	ap.add_argument("-d", "--debug", help="add this if you want to see the debug infos like info about the dataset or the tree",action= 'store_true')
	args = vars(ap.parse_args())

	debug=args["debug"]

	dataset=readDataset()			
	training_data,test_data=divideTheDataset(dataset.data)
	header=dataset.header
	size_ts=len(training_data)
	#the teacher says that this is a good number.
	number_of_classificators=int(10)
	d= len(training_data[0])-1
	if(debug):
		getDatasetInfo(dataset,training_data)
	
	# - - - Calculations - - -
	print("[CALCULATION]\n")
	
	# creates the subsets for the c.v.
	sets=createThesubsets(training_data,number_of_classificators)
	#finds the best values of the different values
	print("size ts: "+str(size_ts))
	
	algorithm=input("\nWhich algorithm do you want to use?\n '1'=CrossValidation,\n '2'= Risk Analysis,\n '3'= Just show the validation and test error of the algorithm with tree from 1 node to 2^(d+2) nodes.\n:")

	if(len(algorithm)>0 and not algorithm==''):
		algorithm=int(algorithm) 
	
	if(algorithm==1):
		tree=learnWithCrossVal(sets, header, 3, 2**(d+2), 15, size_ts)
		if(debug):
			print(tree)
		print("test error: "+ str(lossFunction(test_data,tree)/len(test_data)))
		print()
		
	if(algorithm==2):
		deltaValue=float(input("Choose the value of the delta (range=(0,1]): "))
		isOn=int(input("As |o(h)| do you want to use a O(n) estimation or a O(n*log(d)) estimation? Press '0' for O(n), '1' for O(n*log(d)): "))
		if(isOn==0):
			isOn=True
		else:
			isOn=False
		treeRiskh=learnWithTreeRisk(deltaValue, size_ts, training_data,header, d,isOn,15)
		tree=DecisionTree(training_data,header,treeRiskh)
		if(debug):
			print(tree)
		print("test error: "+ str(lossFunction(test_data,tree)/len(test_data)))
		
	if(algorithm==3):
		graphicWithDifferentParameters(sets,training_data, size_ts, test_data,header, 3, 2**(d+2), 20)
		print()
	print()