import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import argparse
import time
import cv2
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization
from skimage.transform import resize
 
def getPlate(CONFIDENCE = 0.5,PADDINGX = 100,PADDINGY = 50,newW = 512, newH =288):
	filepath  = '/home/bp0017/Documents/hackathon/jesse_data/benchmarks-master/endtoend/us/wts-lg-000051.jpg'
	image = cv2.imread(filepath)#np.asarray(Image.open(filepath).convert("L")) #I like pillow's image opening method better
	orig = image.copy()
	(H, W) = image.shape[:2]
	cv2.imshow("Text Detection", image)
	cv2.waitKey(0)

	#16:9 aspect ratio. must be multiples of 32. Manual
	rW = W / float(newW)
	rH = H / float(newH)
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

	net = cv2.dnn.readNet('/home/bp0017/Documents/east/opencv-text-detection/frozen_east_text_detection.pb') #using EAST text detect

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False) #config from https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames) #forward pass
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	 
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] <CONFIDENCE:
				continue
	 
			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
	 
			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
	 
			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
	 
			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
	 
			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	 
	# loop over the bounding boxes
	cropped_plates = []
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		#cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
		#print((startX, startY), (endX, endY))
		cpy = orig.copy()
		cropped = cpy[startY-PADDINGX:endY+PADDINGX,startX-PADDINGY:endX+PADDINGY]
		x,y,depth = cropped.shape
		if (x and y and depth): #if it's not in the image, don't show it
			#cv2.imshow("cropped", cropped)
			#cv2.waitKey(0)
			cropped_plates.append(cropped) #might be multiple plates in



	# for img in cropped_plates:
	# 	cv2.imshow("Text Detection", img)
	# 	cv2.waitKey(0)

		return cropped_plates

#@param plate_img the localized image of the plate (hopefully)
def get_chars(plate_img):
	gray_image = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) *255
	#thresh = threshold_otsu(gray_image)
	ret,binary = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
	label_image = measure.label(binary)
	fig, (ax1) = plt.subplots(1)
	ax1.imshow(gray_image, cmap="gray");
	#cv2.imshow("binary",thresh)
	#cv2.waitKey(0)
	rectanges = []
	for region in regionprops(label_image):
	    if region.area < 50:
	        #if the region is so small then it's likely not a license plate
	        continue
	    minRow, minCol, maxRow, maxCol = region.bbox
	    rect = (minRow, minCol, maxRow, maxCol,region.area)
	    rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
	    ax1.add_patch(rectBorder)
	    rectanges.append(rect)
	medianArea = np.median([x[4] for x in rectanges])
	selected_chars = []
	for r in rectanges:
		minRow, minCol, maxRow, maxCol,area = r
		if area < 3*medianArea:#welcome to hackathon, where numbers are made up and statistics don't matter
			selected_chars.append(r)

	selected_chars_sorted = sorted(selected_chars, key = lambda x: x[1]) 
	for s in selected_chars_sorted:
		minRow, minCol, maxRow, maxCol,area = s
		cpy = binary.copy()
		sliced = cpy[minRow:maxRow,minCol:maxCol]
		resized_image = cv2.resize(sliced, (15, 25),interpolation=cv2.INTER_NEAREST) 
		cv2.imshow("char",resized_image)
		cv2.waitKey(0)

	#plt.show()


get_chars(getPlate()[0])