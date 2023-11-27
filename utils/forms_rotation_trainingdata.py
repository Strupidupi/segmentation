import os
from PIL import Image
import numpy as np
from get_min_max_dimensions import get_min_max_dimension
from restructure_data import resize_image

CSV_FILE = '/rotation_labels.csv'
DATASET_DIR = '../dataset_rotation'
ROTATIONS_DIR = '/rotations'
SINGLE_FORMS_DIR = DATASET_DIR + '/single_forms/masks'
ROTATION_STEPS = 4

SHAPE_ROTATIONS = {
    'circle': 0,
    'ellipse': 180,
    'heart': 360,
    'parallelogram': 180,
    'pentagon': 72,
    'rectangle': 180,
    'square': 90,
    'star': 72,
    'triangle': 120,
}

min_width, max_height, max_width, max_height = get_min_max_dimension(SINGLE_FORMS_DIR)
new_width = ((max_width // 32) + 1) * 32
new_height = ((max_height // 32) + 1) * 32

def generate_rotation_training_data(image_path, file_components, rotation):
    if not os.path.exists(DATASET_DIR + ROTATIONS_DIR):
        os.makedirs(DATASET_DIR + ROTATIONS_DIR)

    img = resize_image(image_path, new_width=new_width, new_height=new_height)
    img = Image.fromarray(img)
    img_rotate = img.rotate(rotation)
    file_name_rotated = file_components[0] + '_' + str(rotation) + file_components[1]
    file_rotated_path = DATASET_DIR + ROTATIONS_DIR + '/' + file_name_rotated
    img_rotate.save(file_rotated_path)
    return file_name_rotated


def generate_rotations(max_rotation):
    return [rotation for rotation in range(0, max_rotation, ROTATION_STEPS)]


def generate_rotations_for_all_shapes():
    for subdir, dirs, files in os.walk(SINGLE_FORMS_DIR):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            file_path = subdir + '/' + file
            file_components = os.path.splitext(file)
            file_name = file_components[0]
            max_rotation = SHAPE_ROTATIONS.get(file_name.lower())
            rotations = generate_rotations(max_rotation)
            for angle in rotations:
                generate_rotation_training_data(file_path, file_components, angle)


if __name__ == '__main__':
    generate_rotations_for_all_shapes()