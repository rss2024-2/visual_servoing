import cv2
import imutils
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging.
	Press any key to continue.
	"""
	winname = "Image.jpg"
	# cv2.namedWindow(winname)        # Create a named window
	# cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
	#cv2.imshow(winname, img)
	cv2.imwrite(winname, img)
	#cv2.waitKey()
	# cv2.destroyAllWindows()

def cd_sift_ransac(img, template):
	"""
	Implement the cone detection using SIFT + RANSAC algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	# Minimum number of matching features
	image_print(template)
	MIN_MATCH = 8 # Adjust this value as needed
	# Create SIFT
	sift = cv2.xfeatures2d.SIFT_create()

	# Compute SIFT on template and test image
	kp1, des1 = sift.detectAndCompute(template,None)
	kp2, des2 = sift.detectAndCompute(img,None)

	# Find matches
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

	# Find and store good matches
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)

	# If enough good matches, find bounding box
	if len(good) > MIN_MATCH:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	
	
		# Create mask
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()

	

		h, w = template.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

		# Transform the points using the homography matrix
		transformed_pts = cv2.perspectiveTransform(pts, M)
		
		print(transformed_pts.shape)

		# Calculate the bounding box coordinates
		x_min = transformed_pts[0][0][0]
		y_min = transformed_pts[0][0][1]
		x_max = transformed_pts[2][0][0]
		y_max = transformed_pts[2][0][1]

		# Return bounding box
		return ((x_min, y_min), (x_max, y_max))
	else:

		print(f"[SIFT] not enough matches; matches: ", len(good))

		# Return bounding box of area 0 if no match found
		return ((0,0), (0,0))

def cd_template_matching(img, template):
	"""
	Implement the cone detection using template matching algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	template_canny = cv2.Canny(template, 50, 200)

	# Perform Canny Edge detection on test image
	grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_canny = cv2.Canny(grey_img, 50, 200)

	# Get dimensions of template
	(img_height, img_width) = img_canny.shape[:2]

	# Keep track of best-fit match
	best_match = None
	best_scale = None
	max_score = -float('inf')  # Initialize max_score to negative infinity


	# Loop over different scales of image
	for scale in np.linspace(1.5, .5, 50):
		# Resize the image
		resized_template = imutils.resize(template_canny, width = int(template_canny.shape[1] * scale))
		(h,w) = resized_template.shape[:2]
		# Check to see if test image is now smaller than template image
		if resized_template.shape[0] > img_height or resized_template.shape[1] > img_width:
			continue

		########## YOUR CODE STARTS HERE ##########
		# Use OpenCV template matching functions to find the best match
		# across template scales.

		# Remember to resize the bounding box using the highest scoring scale
		# x1,y1 pixel will be accurate, but x2,y2 needs to be correctly scaled

		result = cv2.matchTemplate(img_canny, resized_template, cv2.TM_CCOEFF_NORMED)
		_, max_val, _, max_loc = cv2.minMaxLoc(result)
		if max_val > max_score:
			best_match = result
			best_scale = scale
			max_score = max_val

		# # Find the location of the best match
		# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

		# top_left = max_loc
		# bottom_left = (top_left[0], top_left[1]+h)
		# top_right = (top_left[0] + w, top_left[1])
		# bottom_right = (top_left[0]+w, top_left[1]+h)

		# img_bb = img.copy()

		# bounding_box = (top_left,_right)
		# cv2.rectangle(img_bb,top_left, bottom_right, 255, 2)
		# print(bounding_box)
		# image_print(img_bb)
		########### YOUR CODE ENDS HERE ###########

	(h, w) = template_canny.shape[:2]
	top_left = max_loc
	bottom_right = (int(top_left[0] + w * best_scale), int(top_left[1] + h * best_scale))
	
	bounding_box = (top_left, bottom_right)
	img_bb = img.copy()
	cv2.rectangle(img_bb,top_left, bottom_right, 255, 2)
	image_print(img_bb)
	
	return bounding_box
