from PIL import Image
import os

DATASET_DIR = '../Datensatz Gewebe/Originale mit Maske/'
# DATASET_DIR = '../dataset/'

def get_min_max_dimension(file_dir=DATASET_DIR):
    min_width, min_height = None, None
    max_width, max_height = 0, 0
    for subdir, dirs, files in os.walk(file_dir):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            file_path = subdir + '/' + file
            file_width, file_height = get_img_size(file_path)
            min_width = min(min_width, file_width) if min_width is not None else file_width
            min_height = min(min_height, file_height) if min_height is not None else file_height
            max_width, max_height = max(max_width, file_width), max(max_height, file_height)
    return min_width, max_height, max_width, max_height


def get_img_size(path):
    return Image.open(path).size


if __name__ == '__main__':
    print(get_min_max_dimension())
