import cv2
import numpy as np
from forms_rotation_trainingdata import SHAPE_ROTATIONS
import matplotlib.pyplot as plt

def get_maximum_overlap(not_rotated, rotated, shape):
    # Get the height and width of the image
    height, width = not_rotated.shape[:2]
    max_count = 0
    max_degree = 0
    count_arr = []
    degree_arr = []
    for angle in range(0, SHAPE_ROTATIONS.get(shape), 4):
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1)

        # Apply the rotation to the image
        rotated_original = cv2.warpAffine(not_rotated, rotation_matrix, (width, height))

        # ensure that the images arre binary
        rotated_original = cv2.threshold(rotated_original, 1, 255, cv2.THRESH_BINARY)[1]
        rotated = cv2.threshold(rotated, 1, 255, cv2.THRESH_BINARY)[1]

        # Perform bitwise AND operation
        result = cv2.bitwise_and(rotated_original, rotated)
        # Count the number of pixels with value 1 in the result
        count_ones = np.count_nonzero(result)
        count_arr.append(count_ones)
        degree_arr.append(angle)
        if count_ones > max_count:
            max_count = count_ones
            max_degree = angle

    # Display the image (optional)
    # cv2.imshow('Image', not_rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.plot(degree_arr, count_arr)
    plt.xlabel('Degree')
    plt.ylabel('Intersecion')
    plt.title('Plot')
    plt.savefig('matching_plot.png')

    return max_degree