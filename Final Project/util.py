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

