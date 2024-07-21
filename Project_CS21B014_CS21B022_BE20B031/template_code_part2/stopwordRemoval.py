from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords


class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []

		stop_words = set(stopwords.words('english'))

		for sentence in text:
			seq = [word for word in sentence if word not in stop_words]
			stopwordRemovedText.append(seq)

		return stopwordRemovedText




	