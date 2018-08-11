class Leaf:
	"""
		A Leaf node classifies data.

		This holds a dictionary of class (e.g., "Apple") -> number of times
		it appears in the rows from the training data that reach this leaf.


		TO DO: this will change in a just 1 label (the majority of the data label for this leaf).
	"""

	def __init__(self, rows):
		self.predictions = self.majority_of_labels(self.class_counts(rows))
		self.rows=rows
		
	def class_counts(self, rows):
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
		
		
	def majority_of_labels(self, counts):
		labels=list(counts.keys())
		max_label=0
		for i in range(0,len(labels)):
			if(counts[labels[i]]>counts[labels[max_label]]):
				max_label=i
				
		return labels[max_label]