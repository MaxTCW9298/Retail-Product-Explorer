# Search

from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.explorer import Explorer
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the image of query")
ap.add_argument("-r", "--product-path", required = True,
	help = "Path to the product path")
args = vars(ap.parse_args())

# initialize picture descriptor
cd = ColorDescriptor((8, 12, 3))


# load the query image
query = cv2.imread(args["query"])
features = cd.describe(query)
# search the product
explorer = Explorer(args["index"])
products = explorer.search(features)
# display the query
cv2.imshow("Query", query)
# loop over the results
for (score, productID) in products:
	# display the product image
	product = cv2.imread(args["product_path"] + "/" + productID)
	cv2.imshow("Product", product)
	cv2.waitKey(0)
