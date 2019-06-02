#import data
import csv
import util, math, random, collections
import numpy as np


# The main method
def main():
	data = loadCsvData('all_stocks_5yr.csv')
	data = data[1:]
	stock_dict = sortData(data)
	prev_stock_dict = sortPrevData(data)
	#print prev_stock_dict
	#print len(stock_dict)
	#print len(stock_dict["T"])
	mdp = util.PortfolioMdp(stock_dict, 7, 100)
	#print stock_dict['FIS'][251][4]
	#print mdp.startState()
	
	random_selection = []
	for _ in range(20):
		random_selection.append(generateRandom(stock_dict, 10))

	print np.mean(random_selection)
	print generateLowRisk(stock_dict,10)
	
	
	#print generateRandom(stock_dict, 10)

def generateRandom(stock_dict, num_stocks):
	companies = stock_dict.keys()
	orig_investment = 100
	total_return = orig_investment
	for date in range(252):
		indices = random.sample(range(len(stock_dict)), num_stocks)
		daily_return = 0
		for i in indices: #get the return for each stock on a given day
			company = companies[i]
			daily_return += calculateReward(stock_dict, company, total_return, num_stocks, date)
		#print daily_return
		total_return += daily_return
	return float(total_return)/orig_investment

		
def generateLowRisk(stock_dict, num_stocks):
	companies = stock_dict.keys()
	orig_investment = 100
	total_return = orig_investment
	for date in range(252):
		indices = findLowRisk(stock_dict)
		daily_return = 0
		for i in indices: #get the return for each stock on a given day
			company = companies[i]
			daily_return += calculateReward(stock_dict, company, total_return, num_stocks, date)
		#print daily_return
		total_return += daily_return
	return float(total_return)/orig_investment

def findLowRisk(prev_stock_dict):
	company_stds = []
	for company in prev_stock_dict:
		deltas = []
		for date in range(len(prev_stock_dict[company])):
			open_price = prev_stock_dict[company][date][1]
			close_price = prev_stock_dict[company][date][4]
			if open_price == "" or close_price == "":
				open_price = prev_stock_dict[company][date -1][1]
				close_price = prev_stock_dict[company][date-1][4]
			open_price = float(open_price)	
			close_price = float(close_price)
			delta = close_price - open_price 
			deltas.append(abs(delta))
		std = np.std(deltas)
		company_stds.append(std)
	indices = []
	for _ in range(10):
		i = company_stds.index(min(company_stds))
		indices.append(i)
		company_stds.pop(i)

	return indices



	







def calculateReward(stock_dict, company, reward, num_stocks, date):
	open_price = stock_dict[company][date][1]
	close_price = stock_dict[company][date][4]
	open_price = float(open_price)	
	close_price = float(close_price)
	delta = close_price - open_price 
	investment = float(reward)/num_stocks
	num_shares = float(investment)/open_price

	stock_return = num_shares * delta
	#print stock_return
	return stock_return



	


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
	matrix = np.array(matrix)
	for row in matrix:
		company = row[6]
		if row[0][:4] == '2015':
			if company != previous_company:
				sorted_data[company] = [[row[i] for i in range(6)]]
				previous_company = company
			else:
				if company in sorted_data:
						sorted_data[company].append([row[i] for i in range(6)])

	sorted_data_trimmed = {}
	for data in sorted_data:
		if len(sorted_data[data]) == 252:
			sorted_data_trimmed[data] = sorted_data[data]
	

	return sorted_data_trimmed


def sortPrevData(matrix):
	sorted_data = {}
	previous_company = "default"
	for row in matrix:
		company = row[6]
		if row[0][:4] == '2015':
			break
		if company != previous_company:
			sorted_data[company] = [[row[i] for i in range(6)]]
			previous_company = company
		else:
			if company in sorted_data:
					sorted_data[company].append([row[i] for i in range(6)])
	

	return sorted_data


class QLearningAlgorithm:
	def __init__(self, actions, discount, featureExtractor):
		self.actions = actions
		self.discount = discount
		self.featureExtractor = featureExtractor
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