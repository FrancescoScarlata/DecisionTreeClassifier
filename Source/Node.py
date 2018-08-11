class Node:
	'''

	This class is the node that is created before decide if it is a leaf or it can be splitted in a inner node.
	It needs to hold a reference to the rows, the best gain and question that will have that gain.
	The father reference is because if it is destroyed, has to inform the father of the substitution
	The is a TrueChild, keep Track of wether it is a true child or a false child 
	'''
	def __init__(self, question, gain, rows, father, isATrueChild):
		self.question = question
		self.gain = gain
		self.rows = rows
		self.isATrueChild=isATrueChild
		self.father =father