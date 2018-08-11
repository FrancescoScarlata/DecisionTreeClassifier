class Question:
	"""
	A Question is used to partition a dataset.
	
    This class just records a 'column number' [the number of the column] (e.g., 0 for Color) and a
    'column value' (e.g., Green). 
	The 'match' method is used to compare the feature value in an example 
	to the feature value stored in the question. 
	See the demo below.
    """

	def __init__(self, column, value, header):
		self.column = column
		self.value = value
		self.header=header

	def match(self, example):
		# Compare the feature value in an example to the
		# feature value in this question.
		val = example[self.column]
		if self.is_numeric(val):
			return val >= self.value
		else:
			return val == self.value

	def __repr__(self):
		# This is just a helper method to print
		# the question in a readable format.
		condition = "=="
		if self.is_numeric(self.value):
			condition = ">="
		return "Is %s %s %s?" % (self.header[self.column], condition, str(self.value))
		
	def is_numeric(self, value):
		"""
		Test if a value is numeric.
		Returns true if it is, false otherwise
		"""
		return isinstance(value, int) or isinstance(value, float)