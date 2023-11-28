import cv2
import numpy as np

def get_maximum_overlap(path_to_original_not_rotated_image : str, path_to_rotated_image: str):
    # Read an image from file
    not_rotated = cv2.imread('path/to/your/image.jpg')
    rotated = cv2.imread('path/to/your/image.jpg')

    # Get the height and width of the image
    height, width = not_rotated.shape[:2]
    max_count = 0
    max_degree = 0
    for i in range(90):

        # Define the rotation angle (in degrees)
        angle = i * 4

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        # Apply the rotation to the image
        rotated_original = cv2.warpAffine(not_rotated, rotation_matrix, (width, height))

        # ensure that the images arre binary
        rotated_original = cv2.threshold(rotated_original, 1, 255, cv2.THRESH_BINARY)[1]
        rotated = cv2.threshold(rotated, 1, 255, cv2.THRESH_BINARY)[1]

        # Perform bitwise AND operation
        result = cv2.bitwise_and(rotated_original, rotated)
        # Count the number of pixels with value 1 in the result
        count_ones = np.count_nonzero(result)
        if count_ones > max_count:
            max_count = count_ones
            max_degree = angle

    # Display the image (optional)
    # cv2.imshow('Image', not_rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return angle