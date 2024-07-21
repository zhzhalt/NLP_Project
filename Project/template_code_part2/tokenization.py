from util import *

# Add your import statements here
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer



class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		# converting to lowercase and using split to split based on whitespace 

		for sentence in text:
			tokenizedText.append(sentence.lower().split(' '))

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		t = TreebankWordTokenizer()

		for sentence in text:
			tokenizedText.append(t.tokenize(sentence))

		return tokenizedText