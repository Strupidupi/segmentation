import os
import cv2
import numpy as np
import cv2

colors_tint = [
    (14.915178181182435, 135.71715376322155, 153.73434540633468, 0.0),
    (114.63410256410256, 69.54222222222222, 120.4982905982906, 0.0),
    (46.308826726574395, 107.09602145082978, 157.0666352634249, 0.0),
    (117.47395382395382, 85.25411255411255, 17.592099567099567, 0.0),
    (78.72327258275138, 93.24217907227616, 58.557599500369044, 0.0),
    (52.5188154214366, 68.13610542403184, 148.50915539663774, 0.0),
    (21.61256242796773, 21.117556665386093, 113.54511621206301, 0.0), 
    (50.27987510644337, 121.06112214968304, 98.0293310625414, 0.0),
    (96.005749235474, 55.60954128440367, 41.511151885830785, 0.0),
]
colors = [
    (12.416961130742049, 151.04572498565432, 171.30877955966295, 0.0),
    (137.9826370194896, 65.76343275546462, 133.18641649617746, 0.0),
    (24.476179916472635, 113.32871345677644, 169.43364009313672, 0.0),
    (147.26724746880151, 99.29879915234284, 13.977395808806216, 0.0),
    (76.85377701579385, 94.62829904405653, 31.403392560266003, 0.0),
    (27.253912691295167, 52.34359360083317, 154.8077357016808, 0.0),
    (26.980780222920835, 8.395089073068496, 121.71234638468134, 0.0),
    (50.62677043553923, 138.7696137671001, 102.16184076956543, 0.0),
    (117.24938759383643, 54.606282101935996, 31.742196760173847, 0.0),
]

def get_closest_color_index(color, with_tint): 
    color_list = colors_tint if with_tint else colors
    


# Define the path to the images and masks
path_to_images = './our_data/Angle_Calculation/images'
path_to_masks = os.path.join(path_to_images, 'masks')

# Create the masks directory if it doesn't exist
if not os.path.exists(path_to_masks):
    os.makedirs(path_to_masks)

kernel = np.ones((10, 10), np.uint8) 
x = os.listdir(path_to_images)
x = ['2.jpg', '184c059f-prepared_img_224.jpg']

# Process each file in the directory
for filename in x:
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
    # Load the image
    filepath = os.path.join(path_to_images, filename)
    image = cv2.imread(filepath)
    height,width,_ = image.shape
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold of blue in HSV space 
    upper = np.array([255, 255, 255]) 
    lower = np.array([0, 70, 0])  

    # preparing the mask to overlay 
    mask = cv2.inRange(hsv_image, lower, upper) 
        
    # The black region in the mask has the value of 0, 
    # so when multiplied with original image removes all non-blue regions 
    result = cv2.bitwise_and(image, image, mask = mask) 

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours from mask

    cv2.drawContours(result, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    cv2.imshow("prep", result)
    cv2.waitKey(0) 

    for i,c in enumerate(contours):
        if cv2.contourArea(c) < 1000: continue

        single_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(single_mask, pts=[c], color=255)
        single_mask = cv2.morphologyEx(single_mask, cv2.MORPH_CLOSE, kernel)
        new_filename = f"{i}_{filename}"

        cv2.imwrite(os.path.join(path_to_masks, new_filename), single_mask)
        
        mean_color = cv2.mean(image, mask=single_mask)
        print(mean_color)

    print("Processing complete.")
