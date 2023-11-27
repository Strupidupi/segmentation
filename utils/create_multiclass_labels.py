import cv2
import os

ORIGINAL_DATASET_DIR = "../dataset_forms"

for subdir, dirs, files in os.walk(ORIGINAL_DATASET_DIR):
    files = [f for f in files if not f[0] == '.']
    dirs[:] = [d for d in dirs if not d[0] == '.']
    for file in files:
        filepath = subdir + "/" + file
        print(filepath)
        image = cv2.imread(filepath)