from util import *

# Add your import statements here
import nltk
from nltk.stem import PorterStemmer



class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []

		porter_stemmer = PorterStemmer()
        
		for sentence in text:
			seq = [porter_stemmer.stem(word) for word in sentence]
			reducedText.append(seq)
		
		return reducedText


