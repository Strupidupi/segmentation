from PIL import Image
import os

DATASET_DIR = '../dataset'

def get_max_dimension():
    max_width, max_height = 0, 0
    for subdir, dirs, files in os.walk(DATASET_DIR):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            file_path = subdir + '/' + file
            file_width, file_height = get_img_size(file_path)
            max_width = max(max_width, file_width)
            max_height = max(max_height, file_height)
    return max_width, max_height


def get_img_size(path):
    return Image.open(path).size


if __name__ == '__main__':
    print(get_max_dimension())
