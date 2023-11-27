import os
import shutil
import random
import albumentations as A
import cv2
from get_min_max_dimensions import get_min_max_dimension

# ORIGINAL_DATASET_DIR = '../Datensatz Gewebe/Originale mit Maske'
# NEW_DATASET_DIR = '../dataset'
ORIGINAL_DATASET_DIR = '../dataset_forms/images'
NEW_DATASET_DIR = '../dataset_forms/dataset'

ANNOTATION_DIR_SUFFIX = 'annot'
MASK_SUFFIX = '_mask'
TRAIN = '/train'
VAL = '/val'
TEST = '/test'
DIRS = [TRAIN, VAL, TEST]
LABEL_DIRS = [TRAIN + ANNOTATION_DIR_SUFFIX, VAL + ANNOTATION_DIR_SUFFIX, TEST + ANNOTATION_DIR_SUFFIX]

TRAIN_PORTION = 85
VAL_PORTION = 10
TEST_PORTION = 5
DIR_PORTIONS = [TRAIN_PORTION, VAL_PORTION, TEST_PORTION]

min_width, max_height, max_width, max_height = get_min_max_dimension()
new_width = ((max_width // 32) + 1) * 32
new_height = ((max_height // 32) + 1) * 32

def resize_image(image, new_height=new_height, new_width=new_width):
    img = cv2.imread(image)
    transform = A.PadIfNeeded(min_height=new_height, min_width=new_width, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    augmented_image = transform(image=img)['image']
    return augmented_image

def restructure():
    for new_dir in DIRS + LABEL_DIRS:
        new_path = NEW_DATASET_DIR + new_dir
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        os.makedirs(new_path)

    for subdir, dirs, files in os.walk(ORIGINAL_DATASET_DIR):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            if not MASK_SUFFIX in file:
                file_components = os.path.splitext(file)
                file_name = file_components[0]
                file_ext = file_components[1]
                
                src = subdir + '/' + file
                src_label = subdir + '/' + file_name + MASK_SUFFIX + file_ext
                dst_dir = random.choices(population=DIRS, weights=DIR_PORTIONS, k=1)[0]
                dst = NEW_DATASET_DIR + dst_dir
                dst += '/' + file
                dst_label = NEW_DATASET_DIR + dst_dir + ANNOTATION_DIR_SUFFIX
                dst_label += '/' + file

                augmented_image = resize_image(src)
                augmented_image_label = resize_image(src_label)
                cv2.imwrite(dst, augmented_image)
                cv2.imwrite(dst_label, augmented_image_label)


if __name__ == '__main__':
    restructure()
