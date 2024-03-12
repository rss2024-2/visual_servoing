import cv2
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

lower_orange = np.array([0, 100, 100])
high_orange = np.array([20, 255, 255])
MIN_CONTOUR_AREA = 100

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########
	# Convert the image to HSV
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#use cv2.inRange to apply a mask to the image
	color_mask = cv2.inRange(hsv_img, lower_orange, high_orange)
	img_masked = cv2.bitwise_and(hsv_img,hsv_img, mask=color_mask)
	
	bgr_img_masked = cv2.cvtColor(img_masked, cv2.COLOR_HSV2BGR)

	# Apply thresholding to the masked image
	_, thresholded_image = cv2.threshold(bgr_img_masked, 40, 255, cv2.THRESH_BINARY)
	contours, _  = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) > 0:
                best_contour = max(contours, key=cv2.contourArea) # Choose contour of largest area
                if cv2.contourArea(best_contour) >= MIN_CONTOUR_AREA: # Super small contour --> likely just noise
                    # Build Bounding Box
                    x,y,w,h = cv2.boundingRect(best_contour)
                    bounding_box = ((x,y), (x+w, y+h))
                    return bounding_box
    


	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
