#import data
import csv
import util, math, random, collections
import numpy as np
import matplotlib.pyplot as plt


# The main method
def main():
	data = loadCsvData('all_stocks_5yr.csv')
	data = data[1:]
	stock_dict = sortData(data)

	sp500data = load500Data('sp500_2015.csv')[:30]
	print len(sp500data)
	sp500dates = [row[0] for row in sp500data]
	sp500prices = [row[1] for row in sp500data]

	print "Computing MDP"
	mdp = util.PortfolioMdp(stock_dict, 5, 5)
	print "Computing States"
	dates,_ = mdp.computeStates()
	dates = [d - 4 for d in dates]
	print len(dates)
	print "Executing Q Learning"
	rl = QLearningAlgorithm(mdp.actions, mdp.discount(1), mdp.portfolioFeatureExtractor)
	print "Simulating"
	end_value, gain, values = (util.simulate(mdp,rl,1))
	print len(values)
	print "Portfolio is worth {} and gained {}%". format(end_value, gain)





	generateGraph(dates, values, sp500dates, sp500prices)
	

def generateGraph(x1, y1, x2, y2):
	plt.plot(x1, y1, label = 'Optimal Portfolio')
	plt.plot(x1, y2, label = 'S&P 500')
	plt.ylabel('Portfolio Value')
	plt.xlabel('Days')
	plt.title('Optimal Portfolio vs S&P 500 in January 2015')
	plt.legend()
	plt.show()

	
	
	#print generateRandom(stock_dict, 10)
def baseline():
	random_selection = []
	for _ in range(20):
		random_selection.append(generateRandom(stock_dict, 10))

	print np.mean(random_selection)
	print generateLowRisk(stock_dict,10)

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
			for i in range(len(row)):
				if i == 0 or i == 6 or row[0] == 'date':
					doubleRow.append(row[i])
				else:
					doubleRow.append(float(row[i]))

			# store the row into our matrix
			matrix.append(doubleRow)
	return matrix

def load500Data(fileName):
	matrix = []
	# open a file
	with open(fileName) as f:
		reader = csv.reader(f)
		# loop over each row in the file
		for row in reader:
			# cast each value to a float
			doubleRow = []
			for i in range(len(row)):
				if i == 0:
					doubleRow.append(row[i])
				else:
					value = float(row[i])
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
		if row[0][-4:] == '2015':
			if company != previous_company:
				sorted_data[company] = [[row[i] for i in range(7)]]
				previous_company = company
			else:
				if company in sorted_data:
						sorted_data[company].append([row[i] for i in range(7)])

	print len(sorted_data['T'])
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
		self.weights = collections.defaultdict(lambda: 1.0)
		self.numIters = 0

	# Return the Q function associated with the weights and features
	def getQ(self, state, action):
		score = 0
		for f, v in self.featureExtractor(state, action):
			#print "V:{}" .format(v)
			score += self.weights[str(f)] * v
			print "weights : {}" .format(self.weights[str(f)])
			print "Score:" + str(score)
		print "Action: {}, score: {}" .format(action, score)
		return score

	# This algorithm will produce an action given a state.
	# Here we use the epsilon-greedy algorithm: with probability
	# |explorationProb|, take a random action.
	def getAction(self, state):
		self.numIters += 1
		
		# if random.random() < self.explorationProb:
		# 	return random.choice(self.actions(state))
		# else:
		possible_states = [(self.getQ(state, action), action) for action in self.actions(state)]
		print possible_states 
		best_action = max((self.getQ(state, action), action) for action in self.actions(state))
		if best_action[0] == 0.0:
			return 'Buy'
		return best_action[1]

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
			#print "change {}".format(change)
		for feature in self.featureExtractor(state, action):
			featureName = feature[0]
			featureValue = feature[1]

			change = change * featureValue
			#print "feature value {}".format(featureValue)
			#print "change2 {}".format(change)
			self.weights[str(featureName)] -= change
			#print "weights 2: {}".format(self.weights[str(featureName)])
		# END_YOUR_CODE

def featureExtractor(state, action):
	features = []
	standardDevs = []
	numStocks = len(state[0])
	for stockNumber in range(numStocks):
		variance = 1 #temporary until we figure out how to calculate it
		sumTerm1 = sum([(state[1][stock] ** 2) * variance for stock in state[1]])
		term2 = []
		for otherStockNumber in range(numStocks):
			if (stockNumber != otherStockNumber):
				coVariance = 1 # temporary until we figure out how to calculate it
				term2.append(state[1][stockNumber] * state[1][otherStockNumber] * coVariance)
		sumTerm2 = sum(term2)
		standardDev = math.sqrt(float(sumTerm1) + float(term2))
		standardDevs.append(standardDev)
	features.append(standardDevs)
	return features
	


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
	main()