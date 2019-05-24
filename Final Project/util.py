import collections, random


class PortfolioMdp:
	def __init__(self, stock_dict, N): 
		 	self.stock_dict = stock_dict
			self.N  = N
			self.allocation = []

	def stocks(self, t):
		stocks = []
		for stock in self.stock_dict:
			stocks.append(self.stock_dict[stock][t])
		return stocks

	def startState(self):
		return 

