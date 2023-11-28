import os

import cv2
from get_min_max_dimensions import get_min_max_dimension
from restructure_data import resize_image

# FORMS_DIR = '../dataset_forms/original_images'
# DST_FORMS_DIR = '../dataset_forms/images'
FORMS_DIR = './test_images'
DST_FORMS_DIR = './resized_test_images'

if __name__ == '__main__':
    min_width, max_height, max_width, max_height = get_min_max_dimension(FORMS_DIR)
    new_width = ((max_width // 32) + 1) * 32
    new_height = ((max_height // 32) + 1) * 32
    for subdir, dirs, files in os.walk(FORMS_DIR):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            src = subdir + '/' + file
            dst = DST_FORMS_DIR + '/' + file
            augmented_image = resize_image(src, new_height=new_height, new_width=new_width)
            cv2.imwrite(dst, augmented_image)
