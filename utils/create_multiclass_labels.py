import cv2
import os
import numpy as np

ORIGINAL_DATASET_DIR = "../dataset_forms/images"
ORIGINAL_MASK = "../dataset_forms/masks"

for subdir, dirs, files in os.walk(ORIGINAL_DATASET_DIR):
    files = [f for f in files if not f[0] == '.']
    dirs[:] = [d for d in dirs if not d[0] == '.']
    for file in files:
        masks = {}
        for i in range(9):
            filepath = ORIGINAL_MASK + "/"+ str(i) + "_" + file
            if not os.path.isfile(filepath):
                continue
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            #cv2.imshow('Mask', mask)
            #cv2.waitKey(0)
            masks[i] = mask

        # wende alle masken an
        combined_mask = np.zeros((1216, 1696))
        for idx, mask in masks.items():
            combined_mask[mask == 255] = idx + 1

        
        file_components = os.path.splitext(file)
        file_name = file_components[0]
        file_ext = file_components[1]
        new_folder = "images"
        path = "../dataset_forms/" + new_folder + "/" + file_name + '_mask' + file_ext
        print(path)
        cv2.imwrite(path, combined_mask)
