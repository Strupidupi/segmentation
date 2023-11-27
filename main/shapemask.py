import os
import cv2
import numpy as np
import cv2
from sklearn.cluster import KMeans

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):

        img = self.IMAGE
        
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)

def dominant_colors(image):
    clusters = 10
    dc = DominantColors(result, clusters) 
    colors = dc.dominantColors()
    print(colors)

    blank_image = np.zeros((clusters*30,clusters*30,3), np.uint8)
    for i,c in enumerate(colors):
        blank_image[i*30:(i+1)*30,:] = c
    cv2.imshow("colors", blank_image)
    cv2.waitKey(0) 

# Define the path to the images and masks
path_to_images = './our_data/Angle_Calculation/images'
path_to_masks = os.path.join(path_to_images, 'masks')

# Create the masks directory if it doesn't exist
if not os.path.exists(path_to_masks):
    os.makedirs(path_to_masks)

# Define the HSV ranges for standard colors
color_hsv_ranges = {
    'red': {'lower': (0, 50, 50), 'upper': (10, 255, 255)},
    'green': {'lower': (50, 50, 50), 'upper': (70, 255, 255)},
    'blue': {'lower': (110, 50, 50), 'upper': (130, 255, 255)}
}

kernel = np.ones((5, 5), np.uint8) 

# Define the color for each shape (replace these with your shapes and colors)
shapes_colors = {
    'shape1': 'red',
    'shape2': 'green',
    'shape3': 'blue'
    # Add other shapes and their colors here...
}
x = os.listdir(path_to_images)
# Process each file in the directory
for filename in x:
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
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

        cv2.drawContours(result, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.imshow("prep", result)
        cv2.waitKey(0) 

        for i,c in enumerate(contours):
            if cv2.contourArea(c) < 1000: continue

            single_mask = np.zeros((height,width,3), np.uint8)
            cv2.fillPoly(single_mask, pts=[c], color=(255,255,255))
            single_mask = cv2.dilate(single_mask, kernel, iterations=3)
            new_filename = f"{i}_{filename}"


        # Create and save masks for each shape
        for shape_name, color_name in shapes_colors.items():
            color_range = color_hsv_ranges[color_name]
            lower_val = np.array(color_range['lower'], dtype="uint8")
            upper_val = np.array(color_range['upper'], dtype="uint8")
            mask = cv2.inRange(hsv_image, lower_val, upper_val)

            # Save the mask to the masks folder
            new_filename = f"{shape_name}_{filename}"
            cv2.imwrite(os.path.join(path_to_masks, new_filename), mask)
print("Processing complete.")
