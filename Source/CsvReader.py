import csv

class CsvReader:
	'''
		This class will read and store the dataset and its headers
	'''
	
	def __init__(self,filename_data, filename_header, numericColumns ):
		'''
			This will start the class and read the lines from the file defined in the filename.
			The filename header is the file on which the headers are saved.
			The numeric columns is a vector of indexes of the columns that are numerics, and that need to be converted
		'''
		self.data=self.readLines(filename_data,numericColumns)
		self.header=self.readHeader(filename_header)
				
				
	def readLines(self,filename, numColumns):
		''' This method is will read the all csv file. 
			If there is a numColumns different from 0, than it will convert it to an int 
		'''
		lines=list()
		print("starting the dataset reading")
		if(not numColumns==None):
			with open(filename, newline='') as csvfile:
				reader = csv.reader(csvfile)
				for row in reader:
					for i in range(0, len(numColumns)):
						row[numColumns[i]]=int(row[numColumns[i]])
					lines.append(row)
					#print("row with numbers : "+str(row))
		else:	#no columns to convert
			with open(filename, newline='') as csvfile:
				reader = csv.reader(csvfile)
				for row in reader:
					lines.append(row)
					#print("row: "+str(row))
		print("finished the reading")
		return lines
		
	def readHeader(self, filename):
		''' This method is will read the header from a csv file. '''
		
		lines=list()
		with open(filename, newline='') as csvfile:
				reader = csv.reader(csvfile)
				for row in reader:
					lines=row
					#print("header : "+str(lines))		
		return lines
	