import os
import cv2
import numpy as np
import cv2
    


# Define the path to the images and masks
base_path = './dataset_forms'
path_to_images = os.path.join(base_path, 'images')
path_to_masks = os.path.join(base_path, 'masks')
path_to_labels = os.path.join(base_path, 'labels')

def read_annotations(image_file_name):
    path = os.path.join(path_to_labels, image_file_name.replace(".jpg", ".txt"))
    file = open(path, 'r')
    lines = file.readlines()
    annotations = {}
    for line in lines:
        values = line.split()
        annotations[int(values[0])] = {
            "cx": float(values[1]),
            "cy": float(values[2]),
            "width": float(values[3]),
            "height": float(values[4])
        }
    return annotations

def find_closest_center(mask, annotations):
    height,width = mask.shape
    for key, val in annotations.items():
        cx = int(width * val["cx"])
        cy = int(height * val["cy"])
        if (mask[cy][cx] == 255):
            return key
    return -1

# Create the masks directory if it doesn't exist
if not os.path.exists(path_to_masks):
    os.makedirs(path_to_masks)

kernel = np.ones((10, 10), np.uint8) 
x = os.listdir(path_to_images)

# Process each file in the directory
for filename in x:
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
    # Load the image
    filepath = os.path.join(path_to_images, filename)
    image = cv2.imread(filepath)
    height,width,_ = image.shape
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold of blue in HSV space 
    upper = np.array([255, 255, 255]) 
    lower = np.array([0, 70, 0])  

    # preparing the mask to overlay 
    mask = cv2.inRange(hsv_image, lower, upper) 
        
    # The black region in the mask has the value of 0, 
    # so when multiplied with original image removes all non-blue regions 
    result = cv2.bitwise_and(image, image, mask = mask) 

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours from mask

    #cv2.drawContours(result, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    #cv2.imshow("prep", result)
    #cv2.waitKey(0) 

    annotations = read_annotations(filename)

    for c in contours:
        if cv2.contourArea(c) < 10000: continue

        single_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(single_mask, pts=[c], color=255)
        single_mask = cv2.morphologyEx(single_mask, cv2.MORPH_CLOSE, kernel)

        index = find_closest_center(single_mask, annotations)

        new_filename = f"{index}_{filename}"

        cv2.imwrite(os.path.join(path_to_masks, new_filename), single_mask)

    print("Processing complete.")