#import data
import csv
import util, math, random, collections


# The main method
def main():
	data = loadCsvData('all_stocks_5yr.csv')
	data = sortData(data)
	print len(data)
	printData(data["T"])

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


class QLearningAlgorithm:
	def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

   	def getQ(self, state, action):
   		score = 0
   		for f, v in self.featureExtractor(state, action):
   			score += self.weights[f]*v
   		return score

   	def getAction(self, state):
   		self.numIters += 1
   		return max((self.getQ(state, action), action) for action in self.actions(state))[1]

   	def getStepsize(self):
   		return 1.0/math.sqrt(self.numIters)

   	def incorporateFeedback(self, state, action, reqard, newState):
   		
   		Vopt = 0
        if newState != None:
            Vopt = max((self.getQ(newState, a), a) for a in self.actions(newState))[0]

        prediction = self.getQ(state,action) #float
        target = reward + self.discount*Vopt #float
        phi = self.featureExtractor(state, action) #dictionary
        
        for f, v in self.featureExtractor(state, action):
            self.weights[f] = self.weights[f] - self.getStepSize()*(prediction - target)*v 

def featureExtractor(state, action):
	


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
	main()