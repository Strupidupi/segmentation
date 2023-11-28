import math
import cv2
import numpy as np
import albumentations as A

from maximum_overlap import get_maximum_overlap

NOT_ROTATED_DIR = '../dataset_rotation/single_forms/masks'
# DATA_DIR = '../dataset_rotation/single_forms/masks'
DATA_DIR = '../dataset_forms/masks'

NORMALIZED_VAL = 240
PADDING = 100

def get_center_of_mass(image):
    momentx = 0
    momenty = 0
    count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= 255:
                momentx = momentx + j
                momenty = momenty + i
                count = count + 1

    centx = int(momentx / count)
    centy = int(momenty / count)
    return centx, centy


def create_bounding_box(image, center_of_mass_x, center_of_mass_y):
    x_indices = np.where(np.max(image, axis=0) == 255)
    y_indices = np.where(np.max(image, axis=1) == 255)
    min_x = np.min(x_indices)
    max_x = np.max(x_indices)
    min_y = np.min(y_indices)
    max_y = np.max(y_indices)

    bb_x = max(abs(center_of_mass_x - min_x), abs(center_of_mass_x - max_x))
    bb_y = max(abs(center_of_mass_y - min_y), abs(center_of_mass_y - max_y))
    square_bb_width = max(bb_x, bb_y)
    new_image_radius = int(square_bb_width * math.sqrt(2)) + 1

    image = add_padding(image, 2*new_image_radius, 2*new_image_radius)
    new_center_of_mass_x, new_center_of_mass_y = get_center_of_mass(image)

    return (
        image,
        new_center_of_mass_x - square_bb_width,
        new_center_of_mass_x + square_bb_width,
        new_center_of_mass_y - square_bb_width,
        new_center_of_mass_y + square_bb_width,
    )

def crop_with_bounding_box(image, bb_x1, bb_x2, bb_y1, bb_y2):
    return image[bb_y1:bb_y2, bb_x1:bb_x2]


def add_padding(image, new_height, new_width):
    transform = A.PadIfNeeded(min_height=new_height, min_width=new_width, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return transform(image=image)['image']


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def convert_to_binary(image):
    return cv2.threshold(image,127,255,cv2.THRESH_BINARY)


def prepare_for_matching(image_path):
    image = cv2.imread(image_path, 2)
    _, image = convert_to_binary(image)

    center_of_mass_x, center_of_mass_y = get_center_of_mass(image)
    # image = cv2.circle(image, (center_of_mass_x, center_of_mass_y), radius=0, color=(0, 0, 255), thickness=5)
    image, bb_x1, bb_x2, bb_y1, bb_y2 = create_bounding_box(image, center_of_mass_x, center_of_mass_y)
    # image = cv2.rectangle(image, (bb_x1, bb_y1), (bb_x2, bb_y2), color=(255, 0, 0), thickness=2)
    image = crop_with_bounding_box(image, bb_x1, bb_x2, bb_y1, bb_y2)
    image = normalize(image)
    return image

def normalize(image):
    image = image_resize(image, width=NORMALIZED_VAL, height=NORMALIZED_VAL)
    image = add_padding(image, image.shape[0] + PADDING, image.shape[1] + PADDING)
    _, image = convert_to_binary(image)
    return image

def get_angle(not_rotated_path, rotated_path, shape):
    rotated_image = prepare_for_matching(rotated_path)
    not_rotated_image = prepare_for_matching(not_rotated_path)
    return get_maximum_overlap(not_rotated_image, rotated_image, shape)
    
if __name__ == '__main__':
    TEST_PIC = '/2_90.jpg'
    NOT_ROTATED_PIC = '/Heart.jpg'
    shape = 'heart'
    # TEST_PIC = '/7_87.jpg'
    # NOT_ROTATED_PIC = '/Star.jpg'
    # shape = 'star'
    rotated_image_path = DATA_DIR + TEST_PIC
    not_rotated_image_path = NOT_ROTATED_DIR + NOT_ROTATED_PIC
    angle = get_angle(not_rotated_image_path, rotated_image_path, shape)
    print(angle)
