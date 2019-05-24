#import data
import csv
import util, math, random, collections
import numpy as np


# The main method
def main():
	stock_dict = loadCsvData('all_stocks_5yr.csv')
	stock_dict = sortData(stock_dict)
	#print len(stock_dict)
	#printData(stock_dict["T"])
	mdp = util.PortfolioMdp(stock_dict, 7)
	stocks_t = mdp.stocks(0)
	print len(stocks_t)


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
		if company != previous_company and row[0] == '2013-02-08':
			sorted_data[company] = [[row[i] for i in range(6)]]
			previous_company = company
		else:
			if company in sorted_data:
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

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        change = 0
        if newState != None:
            change = self.getStepSize() * (self.getQ(state, action) - (reward + self.discount * max(self.getQ(newState, action) for action in self.actions(newState))))
        for feature in self.featureExtractor(state, action):
            featureName = feature[0]
            featureValue = feature[1]
            change = change * featureValue
            self.weights[featureName] -= change
        # END_YOUR_CODE


#def featureExtractor(state, action):
	


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
	main()