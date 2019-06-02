import collections, random
import numpy as np


class PortfolioMdp:
	def __init__(self, stock_dict, N, investment): 
		 	self.stock_dict = stock_dict
			self.N  = N #lookback window
			self.allocation = initialize_allocation()
			self.investment = investment
			self.endTime = 100
			self.capital = 10000

	def initialize_allocation():
		allocation = {}
		for stock in self.stock_dict:
			allocation[stock] = 0
		return allocation


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
		state.append(self.capital)

		return state

	def actions(self, state):
		return ['Buy', 'Sell', 'Stay']

	def calculateStockChanges(self, stock):
		open_price = stock[1]
		close_price = stock[4]

		percent_change = (close_price - open_price)/float(open_price)
		
		return percent_change


	def calcReward(self, remaining_capital, allocation):
		total_invested = sum(allocation.values())
		return total_invested + remaining_capital



	#state = [[stock data], {allocation}, current time, capital]
	def succStates(self, state, action):

		if state[2] == self.endTime:
			return []

		results = []

		stocks = state[0][-1]
		new_time = state[2] + 1
		new_data = [self.stocks(t) for t in range(new_time - self.N, new_time)]
		curr_capital = state[3]
		allocation = state[1]

		if action == 'Buy':
			if state[3] > 1000:
				for stock in stocks:
					allocation[stock] += curr_capital * 0.1
					remaining_capital = 0.9 * curr_capital 
					allocation[stock] *= calculateStockChanges(stock)
					newState = [new_data, allocation, new_time, remaining_capital]
					reward = calcReward(remaining_capital, allocation)
					results.append(newState, reward)

		elif action == 'Sell':
			for stock in stocks:
				change = allocation[stock] * 0.1
				allocation[stock] -= change
				allocation[stock] *= calculateStockChanges(stock)
				remaining_capital = curr_capital + change
				newState = [new_data, allocation, new_time, remaining_capital]
				reward = calcReward(remaining_capital, allocation)
				results.append(newState, reward)
 
		else: # Do nothing
			for stock in stocks
				allocation[stock] *= calculateStockChanges(stock)
			newState = [new_data, allocation, new_time, curr_capital]
			results.append(newState)

		return results


	def discount(self, discount):
		return discount

    def computeStates(self):
	    self.states = set()
	    queue = []
	    self.states.add(self.startState())
	    queue.append(self.startState())
	    while len(queue) > 0:
	        state = queue.pop()
	        for action in self.actions(state):
	            for newState, prob, reward in self.succStates(state, action):
	                if newState not in self.states:
	                    self.states.add(newState)
	                    queue.append(newState)

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
	def portfolioFeatureExtractor(state, action):
		#state (stock data for all days, allocation, window, capital)

		def calculateStockSTD(stock, t):
			stock_data = self.stock_dict[stock]

			stock_window = stock_data[t-self.N: t]

			returns_list = []
			for asset in stock_window:
				close_price = asset[4]
				open_price = asset[1]
				returns = close_price - open_price/float(open_price)
				returns_list.append(returns)
			std = np.std(returns_list)
			return std

		def calculateStockCoef(stock1, stock2, t):
			stock1_data = self.stock_dict[stock1]
			stock2_data = self.stock_dict[stock2]

			stock1_window = stock1_data[t-self.N: t]
			stock2_window = stock2_data[t-self.N: t]

			returns1 = [asset[4]-asset[1]/float(asset[1]) for asset in stock1_window]
			returns2 = [asset[4]-asset[1]/float(asset[1]) for asset in stock2_window]

			numerator = 0
			for i in range(len(stock1_window)):
				numerator += ((returns1[i] - np.mean(returns1))*(returns2[i] - np.mean(returns2)))

			return float(numerator)/(self.N - 1)


	    features = []

	    # Standard Deviation
	    totalInvested = float(sum(self.allocation.values()))
	    finalStdDev = 0
	    for stock in self.allocation if self.allocation[stock] > 0:
	    	propOfStock = float(self.allocation[stock])/totalInvested
	    	varianceOfStock = (calculateStockSTD(stock, state[2]))**2
	    	finalStdDev += (propOfStock ** 2) + (varianceOfStock ** 2)

	    for stock in self.allocation if self.allocation[stock] > 0:
	    	for otherStock in self.allocation if self.allocation[stock] > 0:
	    		if stock != otherStock:
	    			propOfStock1 = float(self.allocation[stock])/totalInvested
	    			propOfStock2 = float(self.allocation[otherStock])/totalInvested
	    			correlationCoef = calculateStockCoef(stock, otherStock, state[2])
	    			stdDevStock = calculateStockSTD(stock, state[2])
	    			stdDevOtherStock = calculateStockSTD(otherStock, state[2])
	    			finalStdDev += (propOfStock1 * propOfStock2 * correlationCoef * stdDevStock * stdDevOtherStock)

	    finalStdDev = math.sqrt(finalStdDev)

	    featureKey = (state, action)
	    featureValue = finalStdDev
	    features.append((featureKey, featureValue))

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
	    for stock in self.allocation if self.allocation[stock] > 0:
	    	if stock in industryTypes.keys():
	    		stockIndustry = industryTypes[stock]
	    		scoreBasedOnIndustry = calculateIndustryScore(stockIndustry)
	    		totalIndustryScore += scoreBasedOnIndustry
	    	else:
	    		scoreBasedOnIndustry = 7.11 # Average S&P 500 Movement in the past 12 years
	    		totalIndustryScore += scoreBasedOnIndustry
	    featureIndustryKey = (state, action)
	    featureIndustryValue = totalIndustryScore
	    features.append((featureIndustryKey, featureIndustryValue))


	    return features


	





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





