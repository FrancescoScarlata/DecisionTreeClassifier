'''
At first i'm taking the script from the https://www.youtube.com/watch?v=LDRbO9a6XPU tutorial about the decision tree.
We want to do a CART classifier, but in the end i'd like to do a iterative algorithm instead of a recursive one. We'll see.
'''
from DecisionNode import DecisionNode
from Leaf import Leaf
from Question import Question
from Node import Node
from CsvReader import CsvReader
import numpy as np
import math
import time
from pathlib import Path
import os.path

'''
# toy training data set. This will be deleted when the algoritm works and i've downloaded the right datasets.
# Format: each row is an example.
# The last column is the label.

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
	['Yellow', 4, 'Lemon'],
	['Yellow', 2, 'Lemon'],
]

# Column labels.
# These are used only to print the tree.
header = ["color", "diameter", "label"]
'''

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


def divideInHalfTheDataset(dataset):
	'''
		This method will divide the dataset in 2: test set and training set.
		It will return the training set and the test set.
	'''
	training_set=list()
	test_set=list()
	for i in range (0, len(dataset)):
		if(i%2==0):
			training_set.append(dataset[i])
		else:
			test_set.append(dataset[i])
	
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
		measure=classify(row,tree)
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
	print("[External CV]\nNumber of sets: "+str(len(sets))+", number of nodes: "+str(number_of_nodes)+", size of the training set: "+str(size_ts))
	errors=list()
	training_set=list()
	
	#loop to make len(sets) trees and determine trees and the losses of each validation set.
	for i in range(0, len(sets)):
		for j in range(0,len(sets)):
			if(not i==j):
				training_set+=sets[j]
		#create the tree with N-1 blocks
		tree=iterative_build_tree(training_set,number_of_nodes)
		#determine the error function on the 1 block call validation set
		errors.append(len(sets)/size_ts*lossFunction(sets[i],tree))
	
	error=np.mean(errors)
	print ("\n[External CV]\nThe Mean error is: "+ str(error)+" with the number of nodes: "+ str(number_of_nodes))
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
	print("[INTERNAL CV]\nCalculating between "+str(starting_num_of_nodes)+" to "+str(ending_num_of_nodes)+"=(2^(d+1)) with "+str(different_nodes_values)+" different values in this interval")
	if(starting_num_of_nodes>ending_num_of_nodes):
		print("Error: starting should be lower of the ending one")
	
	if(different_nodes_values<=0):
		print("Error: the different_nodes_values should be >0 (maybe at least 2)")
	
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
	

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])
	
	
def class_counts(rows):
    """
	Counts the number of each type of example in a dataset.
	As it can be imagined, the max count of a label in that node, will give the label to the node.
	"""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts
	
def is_numeric(value):
    """
	Test if a value is numeric.
	Returns true if it is, false otherwise
	"""
    return isinstance(value, int) or isinstance(value, float)

def partition(rows, question):
    """
	Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

	
def gini(rows):
    """
	Calculate the Gini Impurity for a list of rows.
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

	
def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
	
	
def find_best_split(rows,header):
	"""
	Find the best question to ask by iterating over every feature / value
	and calculating the information gain."""
	best_gain = 0  # keep track of the best information gain
	best_question = None  # keep train of the feature / value that produced it
	current_uncertainty = gini(rows)
	n_features = len(rows[0]) - 1  # number of columns

	for col in range(n_features):  # for each feature

		values = unique_vals(rows,col)  # unique values in the column

		for val in values:  # for each value

			question = Question(col, val,header)
            # try splitting the dataset
			true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
			if len(true_rows) == 0 or len(false_rows) == 0:
				continue

            # Calculate the information gain from this split
			gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
			if gain >= best_gain:
				best_gain, best_question = gain, question
	return best_gain, best_question


def iterative_build_tree(rows, maxNodes):
	"""
		Builds the tree in an iterative way.
		It will start with 1 node at the start and has to learn with at max the number of nodes decided.
	
    """
	currentNodes=1 #the first node is the root.
	
	gain,question=find_best_split(rows,header)
	
	# Since we can ask no further questions,
    # we'll return a leaf.
	if gain== 0	:
		return Leaf(rows)	
	else :				# now starts the real problem
		root=Node(question, gain, rows, None,None)	#The root is a node now
		
		true_rows, false_rows= partition(root.rows, root.question)
		
		gainT,questionT = find_best_split(true_rows,header)	#finds the best gini for both the false and the true
		gainF,questionF = find_best_split(false_rows,header)
		
		true_branch=None
		false_branch=None
		nodes_to_split= list()
		
		root= DecisionNode(question,None, None)
		
		if(gainT==0):		# Check if the gain is 0... in that case that's a leaf
			true_branch=Leaf(true_rows)
		else:
			true_branch=Node(questionT,gainT,true_rows, root, True)
			nodes_to_split.append(true_branch)
			
		root.true_branch=true_branch
		if(gainF==0):		# Check if the gain is 0... in that case that's a leaf
			false_branch=Leaf(false_rows)
		else:
			false_branch=Node(questionF,gainF,false_rows,root, False)
			nodes_to_split.append(false_branch)
			
		root.false_branch=false_branch
		
		currentNodes+=2

		#the number of nodes are not the max means that i can still partitionate if there are nodes that allows that
		#if the nodes to split are==0 means that there are not nodes to partitionate
		while(currentNodes<maxNodes and (not len(nodes_to_split)==0)): 
			# find the best gain from all the nodes. should be sorted?
			max=0
			bestNodeIndex=0
			for i in range(0,len(nodes_to_split)):
				
				if(nodes_to_split[i].gain>max):
					max=nodes_to_split[i].gain
					bestNodeIndex=i
			
			#Now that we have the node with the best gain, we should partition as we did with the root
			i=bestNodeIndex

			true_rows, false_rows= partition(nodes_to_split[i].rows, nodes_to_split[i].question)
			gainT,questionT=find_best_split(true_rows,header)	#finds the best gini for both the false and the true
			gainF,questionF=find_best_split(false_rows,header)
						
			if(nodes_to_split[i].isATrueChild):	#the node has to stay on the true_branch
				
				nodes_to_split[i].father.true_branch=DecisionNode(nodes_to_split[i].question,None,None)
				if(gainT==0):
					nodes_to_split[i].father.true_branch.true_branch=Leaf(true_rows)
				else:
					true_branch=Node(questionT,gainT,true_rows, nodes_to_split[i].father.true_branch, True)
					nodes_to_split[i].father.true_branch.true_branch=true_branch
					nodes_to_split.append(true_branch)
				
				if(gainF==0):
					nodes_to_split[i].father.true_branch.false_branch=Leaf(false_rows)
				else:
					false_branch=Node(questionF,gainF,false_rows, nodes_to_split[i].father.true_branch, False)
					nodes_to_split[i].father.true_branch.false_branch=false_branch
					nodes_to_split.append(false_branch)
						
			else:	#the node has to stay on the false_branch of the father
				nodes_to_split[i].father.false_branch=DecisionNode(nodes_to_split[i].question,None,None)
				if(gainT==0):
					nodes_to_split[i].father.false_branch.true_branch=Leaf(true_rows)
				else:
					true_branch=Node(questionT,gainT,true_rows, nodes_to_split[i].father.false_branch, True)
					nodes_to_split[i].father.false_branch.true_branch=true_branch
					nodes_to_split.append(true_branch)
				
				if(gainF==0):
					nodes_to_split[i].father.false_branch.false_branch=Leaf(false_rows)
				else:
					false_branch=Node(questionF,gainF,false_rows, nodes_to_split[i].father.false_branch, False)
					nodes_to_split[i].father.false_branch.false_branch=false_branch
					nodes_to_split.append(false_branch)
			
			del nodes_to_split[i]	#delete the now decision Node from the list of Nodes to split
			currentNodes+=2
		
		''' 
		Now there are 2 cases:
		1) the max number of nodes is reached. 
			This mean that if there are other Nodes in the list, those have to become a leaf.
		2) the length of the node list is 0, this means that there are no other question to ask
		
		We can check those cases with the len of the node list
		'''
		if(len(nodes_to_split)>0):
			for node in nodes_to_split:
				if(node.isATrueChild==True):
					node.father.true_branch=Leaf(node.rows)
				else:
					node.father.false_branch=Leaf(node.rows)
		
		#print("Number of total node (inner included):"+ str(currentNodes))
		return root

	
def build_tree(rows):
    """
	The tutorial tree
	Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.

    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows, header)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return DecisionNode(question, true_branch, false_branch)
	
def print_tree(node, spacing=""):
	"""World's most elegant tree printing function."""

	# Base case: we've reached a leaf
	if isinstance(node, Leaf):
		#print (spacing + "Records:")
		#for row in node.rows:
			#print (spacing,row)
		print (spacing + "Predicts", node.predictions)
		return

	# Print the question at this node
	print (spacing + str(node.question))

	# Call this function recursively on the true branch
	print (spacing + '--> True:')
	print_tree(node.true_branch, spacing + "  ")

	# Call this function recursively on the false branch
	print (spacing + '--> False:')
	print_tree(node.false_branch, spacing + "  ")
		
def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)
		
	
def getDatasetInfo(dataset):
	# - - - Informations - - - 
	print("\n[DATASET INFOS]:\n")
	print("size of the dataset: "+ str(len(dataset.data)))
	print("size of the training set (m): "+ str(int(len(dataset.data)/2)))
	print("number of attributes (d)= "+str(len(dataset.data[0])-1))
	print("\nNodes(N) <<(2^(d))= "+str(int(	2**(len(dataset.data[0])-1)	))+"\n")
	number_of_nodes=int(2**(len(dataset.data[0])-2))
	try:
		bignumber=int((2*(len(dataset.data[0])-2)*math.e)**number_of_nodes)
	except:
		print("possible statistical classificators cardinality is too big to be calculated!")
		bignumber=np.nan
	print("possibile statistical classificators <=(2de)^N = "+str(bignumber)+" where N= 2^(d-1)\n")
	
	
if __name__=='__main__':
	dataset=readDataset()			
	training_data,test_data=divideInHalfTheDataset(dataset.data)
	header=dataset.header
	size_ts=len(training_data)
	#the teacher says that this is a good number.
	number_of_classificators=int(10)
	
	# - - - Calculations - - -
	startTime=time.clock()
	# creates the subsets for the c.v.
	sets=createThesubsets(training_data,number_of_classificators)
	#finds the best values of the different values
	i_star=internal_cross_validation(sets, 3, int(	2**(len(training_data[0]))), 10, size_ts)
	#creates the tree with the values found by the internal cv
	tree=iterative_build_tree(training_data,i_star)
	
	print("The following is the resulting tree:\n")
	print_tree(tree)
	
	print("\nThe test error with the nodes of the tree chosen by the internal cv ("+str(i_star)+")  is: "+str(lossFunction(test_data,tree)))
	
	#creates a tree with N nodes= 2^d
	tree_2d=iterative_build_tree(training_data,2**(len(test_data[0])-1))
	print("\nThe test error with the nodes as 2^(d) ("+str(2**(len(test_data[0])-1))+")  is: "+str(lossFunction(test_data,tree_2d)))
	
	endTime= time.clock()
	print("time to calculate: "+str(int(endTime-startTime))+" secs")
	
	
	
	'''

	tree=iterative_build_tree(training_data,114)

	print("the test error with the tree chosen with the internal cv (114) is: "+str(lossFunction(test_data,tree)))

	tree=iterative_build_tree(training_data,90)

	print("the test error with the tree 90 nodes: "+str(lossFunction(test_data,tree)))
	'''