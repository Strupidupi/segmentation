import os
import cv2
import numpy as np

# Define the path to the images and masks
path_to_images = '../our_data/Angle_Calculation/images'
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

# Define the color for each shape (replace these with your shapes and colors)
shapes_colors = {
    'shape1': 'red',
    'shape2': 'green',
    'shape3': 'blue'
    # Add other shapes and their colors here...
}
x = os.listdir(path_to_images)
x = ['2.jpg']
# Process each file in the directory
for filename in x:
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load the image
        filepath = os.path.join(path_to_images, filename)
        image = cv2.imread(filepath)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
