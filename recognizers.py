# for recognition 
from flask import Flask, request, jsonify , render_template #pip install flask
from werkzeug.utils import secure_filename
import fitz # pip install PyMuPDF
from tempfile import NamedTemporaryFile
from PIL import Image , ImageEnhance, ImageFilter# pip install pillow
from tqdm import tqdm # pip install tqdm
import os
import cv2 # pip install opencv-python opencv-contrib-python
import numpy as np
import base64
import matplotlib.pyplot as plt # pip install matplotlib

from flask_cors import CORS, cross_origin

# for project , from project
import recognizers 
import extractors 
import compFunc

def is_text_page(page):
    blocks = page.get_text("text")  # Extract text blocks
    return len(blocks) > 0  # If there are text blocks, it's a text page

def is_image_page(page):
    # Check for image blocks
    images = page.get_images()
    print('images')
    print(images)
    if len(images) > 0:
        return True
    
    # Check for existence of transparency groups (potential images)
    for block in page.get_blocks("graphics"):
        if block.dict.get("transparency") is not None:
            return True   
    return False  # No clear evidence of images found


def ImgReadable_assessment(img_array):
	
	# Ensure the image is in RGB format
    if img_array.shape[2] == 4:  # If RGBA, convert to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif img_array.shape[2] == 1:  # If grayscale, convert to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    img_arrayGRAY = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_arrayBGR = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    #extractors.display_image(img_arrayGRAY,'gray')
    #extractors.display_image(img_array,'rgb')
    #extractors.display_image(img_arrayBGR,'BGR')

    # Calculate the Laplacian variance
    laplacian_var = laplacionOperation(img_array)
    print(f'Laplacian var: {laplacian_var}')
    
    # calculating the adaptive threshold
    #adaptiveThresholding_var = adaptiveThresholding(img_array)
    #print(f'adaptive threshold var: {adaptiveThresholding_var}')

    # Save the image as a base64 string
    #image_base64 = base64.b64encode(base_image['image']).decode('utf-8')
    #images.append({
    #    'image_index': xref,
    #    'image_base64': image_base64,
    #    'image_format': base_image['ext'],
    #    'laplacian_var': laplacian_var
    #})

    if laplacian_var>50:
    	return True
    else:
    	return False

def laplacionOperation(img_array):
	# Calculate the Laplacian variance
    laplacian_var = 0
    laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
    print(f'Laplacian variance: {laplacian_var}')
    return laplacian_var

def adaptiveThresholding(img_array):
    # algo for adaptive Thresholding
    resultant_var = 0
    imageText = extractors.Img2Text(img_array)

    #test
    return resultant_var
