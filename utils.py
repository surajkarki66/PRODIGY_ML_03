import os
import numpy as np
import requests

from PIL import Image
from io import BytesIO
from resizeimage import resizeimage
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler


def create_features(image):
    img = image.resize((56, 56))
    img_arr = np.array(img)
    # flatten three channel color image
    color_features = img_arr.flatten()
    # convert image to greyscale
    grey_image = rgb2gray(img_arr)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(8, 8))
    # combine color and hog features into a single array
    flat_features = np.hstack((color_features, hog_features))
    return flat_features

def process_input(input_img_path):
    img = None
    try:
        if input_img_path.startswith('http'):
            response = requests.get(input_img_path)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(input_img_path)
            
    except FileNotFoundError:
        print(f"The image at '{input_img_path}' does not exist.")
        exit(1)
    
    img_feature = create_features(img)
    ss = StandardScaler()
    imgs_stand = ss.fit_transform(img_feature.reshape(1, -1))
    return imgs_stand

