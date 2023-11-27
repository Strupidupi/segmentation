import os
import shutil
import random

ORIGINAL_DATASET = '/Datensatz Gewebe'
ORIGINAL_DATASET_ANNOTATED = '/Originale mit Maske'
NEW_DATASET = '/dataset'
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


def restructure():
    ORIGINAL_DATA_DIR = '.' + ORIGINAL_DATASET
    NEW_DATASET_DIR = '.' + NEW_DATASET
    for new_dir in DIRS + LABEL_DIRS:
        new_path = NEW_DATASET_DIR + new_dir
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        os.makedirs(new_path)

    for subdir, dirs, files in os.walk(ORIGINAL_DATA_DIR + ORIGINAL_DATASET_ANNOTATED):
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
                shutil.copy(src, dst)
                shutil.copy(src_label, dst_label)


if __name__ == '__main__':
    restructure()
