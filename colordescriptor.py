# Color Descriptor

import cv2
import imutils
import numpy as np

class ColorDescriptor
        def __init__(identity, bins):
                # store the number of bins for the 3D histogram
		identity.bins = bins
	def detail(identity, image):
		# convert the image to the HSV color space and initialize the features used to quantify the image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		# compute the center of the image
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))

                # divide the image into four rectangles/segments (top-left, top-right, bottom-right, bottom-left)
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]
		# construct an elliptical mask representing the center of the image
		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
		# loop over the segments
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the image, subtracting  the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)
			# extract a color histogram from the image, then update the feature vector
			hist = identity.histogram(image, cornerMask)
			features.extend(hist)
		# extract a color histogram from the elliptical region and
		# update the feature vector
		hist = identity.histogram(image, ellipMask)
		features.extend(hist)
		# return the feature vector
		return features

	def histogram(identity, image, mask):
		# extract a 3D color histogram from the masked region of the image, using the supplied number of bins per channel
		hist = cv2.calcHist([image], [0, 1, 2], mask, identity.bins,
			[0, 180, 0, 256, 0, 256])
		# normalize the histogram 
		if imutils.is_cv2():
			hist = cv2.normalize(hist).flatten()
		else:
			hist = cv2.normalize(hist, hist).flatten()
		# return the histogram
		return hist

