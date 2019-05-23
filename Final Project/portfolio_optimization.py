#import data


# A useful library for reading files
# with "comma seperated values".
import csv

# The main method
def main():
	data = loadCsvData('all_stocks_5yr.csv')
	data = sortData(data)
	print len(data)
	printData(data["AAL"])

# Reads a files into a 2d array. There are
# other ways of doing this (do check out)
# numpy. But this shows 
def loadCsvData(fileName):
	matrix = []
	# open a file
	with open(fileName) as f:
		reader = csv.reader(f)

		# loop over each row in the file
		for row in reader:

			# cast each value to a float
			doubleRow = []
			for value in row:
				doubleRow.append(value)

			# store the row into our matrix
			matrix.append(doubleRow)
	return matrix

# Prints out a 2d array
def printData(matrix):
	for row in matrix:
		print row

def sortData(matrix):
	sorted_data = {}
	previous_company = "default"
	for row in matrix:
		company = row[6]
		if company != previous_company:
			sorted_data[company] = [[row[i] for i in range(6)]]
			previous_company = company
		else:
			sorted_data[company].append([row[i] for i in range(6)])

	return sorted_data



# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
	main()