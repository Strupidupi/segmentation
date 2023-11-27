import cv2 as cv
import os
import numpy as np

path_to_images = '../dataset/train'
path_dest = '../dataset/train_corrected'
x = os.listdir(path_to_images)

def get_color_correction(image):
	hsv_white_part = cv.cvtColor(image, cv.COLOR_BGR2HSV) 

	upper_white = np.array([255, 125, 255]) 
	lower_white = np.array([0, 0, 125])  
	mask_white_part = cv.inRange(hsv_white_part, lower_white, upper_white)

	img_with_mask = image[np.where(mask_white_part == 255)]
	color_correction = np.array([255, 255, 255] - np.max(img_with_mask, axis=0)).astype(np.uint8)
	print(color_correction, np.max(img_with_mask, axis=0).dtype, color_correction.dtype)
	return color_correction

for filename in x:
	path = os.path.join(path_to_images, filename)
	img = cv.imread(cv.samples.findFile(path))
	img = img + get_color_correction(img)
	cv.imwrite(os.path.join(path_dest, f'cor_{filename}'), img)