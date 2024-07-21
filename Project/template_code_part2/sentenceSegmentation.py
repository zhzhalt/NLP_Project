from util import *

# Add your import statements here
import nltk
from nltk import tokenize



class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []

		previous_end = 0

		# exceptions that follow a '.' that should not cause segmentation

		exceptions = ["Mr", "Ms", "rs"]

		# segmentation should not happen in case of abbreviations

		abbreviations = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

		# punctuation marks that are considered to be the enders for sentences

		enders = ".?!"

		for i in range(len(text)):
			if text[i] in enders: 
				if i+1 != len(text) and text[i+1] in abbreviations: # handling abbreviations
					continue
				if text[i-2:i] not in exceptions: # handling exceptions
					segmentedText.append(text[previous_end:i])
					previous_end = i+1
					continue
				if i+1 != len(text) and i+2 != len(text) and text[i+1] != "." and text[i+2] != "." and text[i-2:i] not in exceptions: # handling ellipses
					segmentedText.append(text[previous_end:i])
					previous_end = i+1

		if previous_end < len(text):
			segmentedText.append(text[previous_end:])

		segmentedText = [sentence for sentence in segmentedText if sentence != "."] # getting rid of '.' if they are also segmented 

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		segmentedText = tokenize.sent_tokenize(text)
		
		return segmentedText