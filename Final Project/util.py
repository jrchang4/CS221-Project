import collections, random, math, csv, copy
import numpy as np
import portfolio_optimization as po



class PortfolioMdp:
	def __init__(self, stock_dict, N, sample): 
			self.stock_dict = stock_dict
			self.N  = N #lookback window

			def initialize_allocation():
				allocation = {}
				for stock in self.stock_dict:
					allocation[stock] = 0
				return allocation
			self.allocation = initialize_allocation()
			self.endTime = 35
			self.capital = 2044.81
			self.sample = sample




	#Returns the stock data for a given day, t
	def stocks(self, t):
		stocks = []
		for stock in self.stock_dict:
			stocks.append(self.stock_dict[stock][t])

		return stocks

	def startState(self):
		state = []

		state.append([self.stocks(t) for t in range(self.N)]) #stock data for previous N days
		state.append(self.allocation) #dictionary
		state.append(self.N) #start time at N
		state.append(self.capital) #uninvested capital
		#state.append(self.capital) #previous reward

		return state

	def actions(self, state):
		return ['Buy', 'Sell', 'Stay']

	def calculateStockChanges(self, stock):
		open_price = float(stock[1])
		close_price = float(stock[4])


		percent_change = close_price/float(open_price)
		
		return percent_change


	def calcReward(self, remaining_capital, allocation):
		total_invested = sum(allocation.values())
		current_value = total_invested + remaining_capital
		#print "Invested: {}" .format(total_invested)
		#print "Uninvested: {}"  .format(remaining_capital)
		# return current_value - previous_value
		# if (current_value > 10138):
		# 	print (current_value)
		return current_value



	#state = [[stock data], {allocation}, current time, capital]
	def succStates(self, state, action):

		if state[2] == self.endTime:
			print "base case"
			return []

		results = []

		stocks = random.sample(state[0][-1],self.sample)
		new_time = state[2] + 1
		new_data = [self.stocks(t) for t in range(new_time - self.N, new_time)]
		curr_capital = state[3]
		allocation = state[1]
		#print "Allocation sum: {}".format(sum(allocation.values()))
		
		# previous_reward = state[4]

		if action == 'Buy':
			if state[3] > 0:
				remaining_capital = 0.9 * curr_capital
				allocation = copy.deepcopy(state[1])

				for stock in stocks:
					allocation[stock[6]] += curr_capital * 0.02
					#print "Allocation sum: {}".format(sum(allocation.values()))

					#print(sum(allocation.values()))
					for stock_name in state[0][-1]:
						allocation[stock_name[6]] *= self.calculateStockChanges(stock_name) 
					#print(sum(allocation.values()))
					#print allocation
					current_value = sum(allocation.values()) + remaining_capital

				newState = [new_data, allocation, new_time, remaining_capital]
				reward = self.calcReward(remaining_capital, allocation)
				#print "Reward: {}".format(reward)
				results.append((newState, reward))

		elif action == 'Sell':
			remaining_capital = curr_capital
			allocation = copy.deepcopy(state[1])
			for stock in allocation:
				change = allocation[stock] * 0.1
				allocation[stock] -= change
				for stock_name in state[0][-1]:
					allocation[stock_name[6]] *= self.calculateStockChanges(stock_name)
				remaining_capital += change
				current_value = sum(allocation.values()) + remaining_capital
			newState = [new_data, allocation, new_time, remaining_capital]
			reward = self.calcReward(remaining_capital, allocation)
			#print "Reward: {}".format(reward)
			results.append((newState, reward))
 
		else: # Do nothing
			allocation = copy.deepcopy(state[1])
			for stock_name in state[0][-1]:
				allocation[stock_name[6]] *= self.calculateStockChanges(stock_name)

			current_value = sum(allocation.values()) + curr_capital
			newState = [new_data, allocation, new_time, curr_capital]
			reward = self.calcReward(curr_capital, allocation)
			results.append((newState, reward))

		#print "num succ states {}" .format(len(results))
		return results


	def discount(self, discount):
		return discount

	def computeStates(self):
		self.states = []
		queue = []
		self.states.append(self.startState())
		queue.append(self.startState())
		dates = []
		max_rewards = []
		while len(queue) > 0:
			#print len(queue)
			state = queue.pop()
			print "Date: {}" .format(state[2])

			rewards = []
			for action in self.actions(state):
				#if len(self.succStates(state, action)) ==  0:
					#print "next in queue"
				for newState, reward in self.succStates(state, action):
					rewards.append(reward)

					if newState not in self.states and newState:
						#print "added"
						self.states.append(newState)
						queue.append(newState)

			if rewards:
				if state[2] == self.endTime - 1:
					break	
				dates.append(state[2])
				max_rewards.append(max(rewards))
				#print "Date: {}, Reward: {}" .format(state[2], max(rewards))
		dates.append(self.endTime)
		return dates, max_rewards

# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
	def portfolioFeatureExtractor(self, state, action):
		#state (stock data for all days, allocation, window, capital)

		def calculateStockSTD(stock, t):
			stock_data = self.stock_dict[stock]

			stock_window = stock_data[t-self.N: t]

			returns_list = []
			for asset in stock_window:
				close_price = float(asset[4])
				open_price = float(asset[1])
				returns = (close_price - open_price)/float(open_price)
				returns_list.append(returns)
			std = np.std(returns_list)
			return std

		def calculateStockCoef(stock1, stock2, t):
			stock1_data = self.stock_dict[stock1]
			stock2_data = self.stock_dict[stock2]

			stock1_window = stock1_data[t-self.N: t]
			stock2_window = stock2_data[t-self.N: t]

			returns1 = [float(asset[4])-float(asset[1])/float(asset[1]) for asset in stock1_window]
			returns2 = [float(asset[4])-float(asset[1])/float(asset[1]) for asset in stock2_window]

			numerator = 0
			for i in range(len(stock1_window)):
				numerator += ((returns1[i] - np.mean(returns1))*(returns2[i] - np.mean(returns2)))

			return float(numerator)/(self.N - 1)

		if len(self.succStates(state,action)) == 0:
			return [((state, action),0)]

		newState = self.succStates(state, action)[0][0]
		features = []

		# Standard Deviation		
		allocation = newState[1]
		totalInvested = float(sum(allocation.values()))
		#print "total invested {}" . format(totalInvested)
		finalStdDev = 0

		for stock in allocation:
			if allocation[stock] > 0:
				#print stock
				propOfStock = float(allocation[stock])/totalInvested
				varianceOfStock = (calculateStockSTD(stock, newState[2]))**2
				finalStdDev += (propOfStock ** 2) + (varianceOfStock)
				#print "finalStdDev {}". format(finalStdDev)

		for stock in allocation: 
			if allocation[stock] > 0:
				for otherStock in allocation: 
					if allocation[otherStock] > 0:
						if stock != otherStock:
							propOfStock1 = float(allocation[stock])/totalInvested
							propOfStock2 = float(allocation[otherStock])/totalInvested
							correlationCoef = calculateStockCoef(stock, otherStock, newState[2])
							stdDevStock = calculateStockSTD(stock, newState[2])
							stdDevOtherStock = calculateStockSTD(otherStock, newState[2])
							finalStdDev += (propOfStock1 * propOfStock2 * correlationCoef * stdDevStock * stdDevOtherStock)

		finalStdDev = math.sqrt(finalStdDev)

		featureKey = (state, action)
		stdValue = -finalStdDev
		#print "Feature value: {}". format(featureValue)

		remaining_capital = newState[3]
		#print "Remaining capital: {}".format(remaining_capital)
		#print "Allocation: {}".format(sum(allocation.values()))

		momentum_value = self.calcReward(remaining_capital, allocation)
		#print "Momentum: {}".format(momentum_value)

		# Industry Type

		def calculateIndustryScore(stockIndustry):
			score = 0
			if stockIndustry == "Consumer Discretionary":
				score += 9.92
			elif stockIndustry == "Consumer Staples":
				score += 8.73
			elif stockIndustry == "Energy":
				score += 1.78
			elif stockIndustry == "Financials":
				score += 0.21
			elif stockIndustry == "Health Care":
				score += 10.31
			elif stockIndustry == "Industrials":
				score += 6.88
			elif stockIndustry == "Information Technology":
				score += 11.18
			elif stockIndustry == "Materials":
				score += 5.51
			elif stockIndustry == "Real Estate":
				score += 4.26
			elif stockIndustry == "Communication":
				score += 4.02
			elif stockIndustry == "Utilities":
				score += 7.16
			return score
		
		grid = loadCsvData('sp500.csv')
		grid = grid[1:]
		industryTypes = {}
		for i in range(len(grid)):
			industryTypes[grid[i][0]] = grid[i][3]

		totalIndustryScore = 0
		for stock in allocation:
			if allocation[stock] > 0:
				if stock in industryTypes.keys():
					stockIndustry = industryTypes[stock]
					scoreBasedOnIndustry = calculateIndustryScore(stockIndustry)
					totalIndustryScore += scoreBasedOnIndustry
				else:
					scoreBasedOnIndustry = 7.11 # Average S&P 500 Movement in the past 12 years
					totalIndustryScore += scoreBasedOnIndustry
		
		featureIndustryKey = (state, action)
		industryValue = totalIndustryScore
		#print "Feature value industry: {}". format(featureValue)

		totalFeatureValue = 10*stdValue + industryValue
		features.append((featureIndustryKey, totalFeatureValue))
		
		#print "Features: {}".format(totalFeatureValue)
		return features


def simulate(mdp, rl, numTrials=10, maxIterations=1000, verbose=False,
			 sort=False):


	totalRewards = []  # The rewards we get on each trial
	totalPortfolioValues = []
	for trial in range(numTrials):
		#print "Trial: {}".format(trial + 1)
		state = mdp.startState()
		sequence = [state]
		totalDiscount = 1
		reward = 0
		counter = 0
		portfolio_values = []
		for i in range(maxIterations):
			counter += 1
			action = rl.getAction(state)
			#print action
			transitions = mdp.succStates(state, action)
			#print "num transitions: {}".format(len(transitions))
			if sort: transitions = sorted(transitions)
			if len(transitions) == 0:
				rl.incorporateFeedback(state, action, 0, None)
				break

			# Choose a random transition
			transition = random.choice(transitions)
			newState, reward = transition
			sequence.append(action)
			sequence.append(reward)
			sequence.append(newState)

			rl.incorporateFeedback(state, action, reward, newState)
			cummGainLoss = (reward - mdp.capital)
			print "Day: {}".format(state[2])
			# print action
			# print "Portfolio value {}" .format(reward)
			# print "Remaining capital {}". format(state[3])
			# print "Cumulative Gain/loss {}".format(cummGainLoss)
			portfolio_values.append(reward)
			totalDiscount *= mdp.discount(1)
			state = newState
		if verbose:
			print "Trial %d (totalReward = %s): %s" % (trial, reward, sequence)
		totalRewards.append(reward)
		totalPortfolioValues.append(portfolio_values)
	return max(totalRewards), float(max(totalRewards)/mdp.capital - 1) * 100, totalPortfolioValues[totalRewards.index(max(totalRewards))]




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

