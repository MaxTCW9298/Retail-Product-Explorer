# Explorer

import numpy as np
import csv

class Explorer:
	def __init__(identity, indexPath):
		# store index path
		identity.indexPath = indexPath
	def search(identity, queryFeatures, limit = 10):
		# initialize our dictionary of products
		products = {}

		# open the index file for reading
		with open(identity.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)
			# loop over the rows in the index
			for row in reader:
				# parse out the image ID and features, then compute the chi-squared distance between the features in our indexand our query features
				features = [float(x) for x in row[1:]]
				d = identity.chi2_distance(features, queryFeatures)
				# udpate products dictionary 
				# the key is the current product ID in the index and the value is the computed distance
				products[row[0]] = d
			# close the reader
			f.close()
		# sort the product where relevant product are at the front of the list
		products = sorted([(v, k) for (k, v) in products.items()])
		# return limited products
		return products[:limit]

	def chi2_distance(identity, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
		# return the chi-squared distance
		return d


