from util import *

# Add your import statements here
import numpy as np
import json
import statistics
with open('cranfield/cran_qrels.json') as f:
	qrels = json.load(f)


# Precision@k - Precision at k is the proportion of relevant documents among the top k retrieved documents.
#
# Recall@k - Recall at k is the proportion of relevant documents that were retrieved among all relevant documents.
#
# F0.5score@k - F0.5 score at k is the weighted harmonic mean of precision and recall, favoring precision. It is calculated using the formula: 
#
# 							(1 + beta^2) * precision * recall
#				F0.5score = ---------------------------------
#							  (beta^2 * precision + recall)
#
#AP@k - Average precision at k is the average of precision values obtained at each relevant document's rank up to k.
#
#nDCG@k - Normalized Discounted Cumulative Gain at k is a measure of ranking quality. It considers both the relevance and ranking position of retrieved documents.

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		# num_retrieved_docs = len(query_doc_IDs_ordered)
		# if num_retrieved_docs >= k:
		num_true_docs_retrieved = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				num_true_docs_retrieved += 1
		precision = num_true_docs_retrieved/k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		precisions = []

		if len(doc_IDs_ordered) == len(query_ids):
			for i in range(len(query_ids)):
				query_doc_IDs_ordered = doc_IDs_ordered[i]
				query_id = query_ids[i]
				true_doc_IDs = []

				for qrel in qrels:
					if int(qrel['query_num']) == int(query_id):
						true_doc_IDs.append(int(qrel['id']))

				precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
				precisions.append(precision)
		
		if len(precisions) > 0:
			meanPrecision = sum(precisions)/len(precisions)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		# num_retrieved_docs = len(query_doc_IDs_ordered)
		num_true_docs = len(true_doc_IDs)
		num_true_docs_retrieved = 0
		# if num_retrieved_docs >= k:
		for i in range(k):
			if int(query_doc_IDs_ordered[i]) in true_doc_IDs:
				num_true_docs_retrieved += 1
		recall = num_true_docs_retrieved/num_true_docs
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		recalls = []

		if len(doc_IDs_ordered) == len(query_ids):
			for i in range(len(query_ids)):
				query_doc_IDs_ordered = doc_IDs_ordered[i]
				query_id = query_ids[i]
				true_doc_IDs = []

				for qrel in qrels:
					if int(qrel['query_num']) == int(query_id):
						true_doc_IDs.append(int(qrel['id']))

				recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
				recalls.append(recall)

		if len(recalls) > 0:
			meanRecall = sum(recalls)/len(recalls)
			
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		fscore = statistics.harmonic_mean([precision, recall])
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		fscores = []
		if len(doc_IDs_ordered) == len(query_ids):
			for i in range(len(query_ids)):
				query_doc_IDs_ordered = doc_IDs_ordered[i]
				query_id = query_ids[i]
				true_doc_IDs = []

				for qrel in qrels:
					if int(qrel['query_num']) == int(query_id):
						true_doc_IDs.append(int(qrel['id']))

				fscore = self.queryFscore(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
				fscores.append(fscore)
		
		if len(fscores) > 0:
			meanFscore = sum(fscores)/len(fscores)
		return meanFscore
	
	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		relevance_vals = {}
		relevant_docs = []

		dcg = 0.0
		idcg = 0.0
		nDCG = 0

		for dict in qrels:
			for doc_id in true_doc_IDs:
				if int(dict['id']) == doc_id and int(dict['query_num']) == query_id:
					relevance_vals[doc_id] = 5 - int(dict['position'])
					relevant_docs.append(int(doc_id))

		for i in range(k):
			doc_id = int(query_doc_IDs_ordered[i])
			if doc_id in relevant_docs:
				relevance = relevance_vals[doc_id]
				dcg = dcg + (relevance/math.log2(i + 2))

		optimal_order = sorted(relevance_vals.values(), reverse = True)
		num_relevant_docs = len(optimal_order)

		num_docs_idcg = min(num_relevant_docs, k)
		for i in range(num_docs_idcg):
			relevance = optimal_order[i]
			idcg = idcg + (relevance/math.log2(i + 2))

		if idcg != 0:
			nDCG = dcg/idcg

		return nDCG



	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		nDCGs = []
		if len(doc_IDs_ordered) == len(query_ids):
			for i in range(len(query_ids)):
				query_doc_IDs_ordered = doc_IDs_ordered[i]
				query_id = query_ids[i]
				true_doc_IDs = []
				for dict in qrels:
					if int(dict['query_num']) == query_id:
						true_doc_IDs.append(int(dict['id']))
				nDCG = self.queryNDCG(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
				nDCGs.append(nDCG)

		if len(nDCGs) > 0:
			meanNDCG = sum(nDCGs)/len(nDCGs)
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""
		avgPrecision = -1

		#Fill in code here
		# num_retrieved_docs = len(query_doc_IDs_ordered)
		# if num_retrieved_docs >= k:
		precisions = []
		i = 0
		for i in range(k):
			if (query_doc_IDs_ordered[i] in true_doc_IDs):
				precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i+1)
				precisions.append(precision)
		if len(precisions) == 0:
			return 0
		avgPrecision = np.mean(precisions)
		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""
		meanAveragePrecision = -1

		#Fill in code here
		avgPrecisions = []
		if len(doc_IDs_ordered) == len(query_ids):
			for i in range(len(query_ids)):
				query_doc_IDs_ordered = doc_IDs_ordered[i]
				query_id = query_ids[i]
				true_doc_IDs = []
				for dict in qrels:
					if int(dict['query_num']) == query_id:
						true_doc_IDs.append(int(dict['id']))
				avgPrecision = self.queryAveragePrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
				avgPrecisions.append(avgPrecision)
		
		if len(avgPrecisions) > 0:
			meanAveragePrecision = np.mean(avgPrecisions)

		return meanAveragePrecision