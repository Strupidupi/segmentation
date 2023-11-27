import os
import cv2
import numpy as np
import cv2
from sklearn.cluster import KMeans

def find_shapes(image):
    # Function to check if a pixel is not black and not already visited
    def is_valid_pixel(x, y, visited):
        return 0 <= x < image.shape[1] and 0 <= y < image.shape[0] and image[y, x] != 0 and not visited[y, x]

    # Function to perform DFS to find all pixels belonging to the same shape
    def dfs(x, y, visited, shape):
        # Directions for neighboring pixels (up, down, left, right)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        # Mark the current pixel as visited and add to the shape
        visited[y, x] = True
        shape.append((x, y))

        # Visit all neighboring pixels
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid_pixel(nx, ny, visited):
                dfs(nx, ny, visited, shape)

    # List to store all shapes
    shapes = []

    # Create an array to mark visited pixels
    visited = np.zeros_like(image, dtype=bool)

    # Iterate through each pixel in the image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if is_valid_pixel(x, y, visited):
                # Found a new shape, perform DFS
                shape = []
                dfs(x, y, visited, shape)
                shapes.append(shape)

    return shapes

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

# Define the path to the images and masks
path_to_images = './our_data/Angle_Calculation/images'
path_to_masks = os.path.join(path_to_images, 'masks')

# Create the masks directory if it doesn't exist
if not os.path.exists(path_to_masks):
    os.makedirs(path_to_masks)

kernel = np.ones((10, 10), np.uint8) 
x = os.listdir(path_to_images)
x = ['2.jpg', '184c059f-prepared_img_224.jpg']

# Process each file in the directory
for filename in x:
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load the image
        filepath = os.path.join(path_to_images, filename)
        image = cv2.imread(filepath)
        
        y = find_shapes(image)
        print('y', y)
        break
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

        single_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(single_mask, pts=[c], color=255)
        single_mask = cv2.morphologyEx(single_mask, cv2.MORPH_CLOSE, kernel)
        new_filename = f"{i}_{filename}"

        cv2.imwrite(os.path.join(path_to_masks, new_filename), single_mask)
        
        mean_color = cv2.mean(image, mask=single_mask)
        print(mean_color)

    print("Processing complete.")
