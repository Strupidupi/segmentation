import os
import cv2
import numpy as np
import cv2
from sklearn.cluster import KMeans


def find_shapes(image):
    """
    Find all connected shapes in an image where the background is black.

    :param image: Input image with shapes
    :return: A list of shapes, where each shape is a list of pixel coordinates
    """
    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize variables
    visited = np.zeros_like(image, dtype=bool)  # To track visited pixels
    shapes = []

    # Define the eight possible directions (neighbors)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    def is_valid(x, y):
        """ Check if a pixel is within the image bounds and not black """
        return 0 <= x < image.shape[1] and 0 <= y < image.shape[0] and image[y, x] != 0

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if is_valid(x, y) and not visited[y, x]:
                stack = [(x, y)]
                shape = []

                while stack:
                    px, py = stack.pop()
                    if is_valid(px, py) and not visited[py, px]:
                        visited[py, px] = True
                        shape.append((px, py))

                        # Add neighbors to stack
                        for dx, dy in directions:
                            nx, ny = px + dx, py + dy
                            if is_valid(nx, ny) and not visited[ny, nx]:
                                stack.append((nx, ny))

                if shape:
                    shapes.append(shape)

    return shapes

def create_shape_images(image, shapes):
    """
    Create and display images for each shape found in the original image.

    :param image: Original image from which shapes were extracted
    :param shapes: List of shapes, where each shape is a list of pixel coordinates
    """
    for i, shape in enumerate(shapes):
        # Create a blank image with the same dimensions as the original
        shape_image = np.zeros_like(image)

        # Draw the shape on the blank image
        for x, y in shape:
            shape_image[y, x] = 255  # Assuming shapes are white on black background

        # Display the shape image
        cv2.imshow(f'Shape {i+1}', shape_image)
        cv2.waitKey(0)  # Waits for a key press to move to the next image

    cv2.destroyAllWindows()

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
path_to_images = '../our_data/Angle_Calculation/images'
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
        
        y = find_shapes(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        create_shape_images(image, y)
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
