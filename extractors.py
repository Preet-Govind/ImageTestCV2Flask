# for extraction
from flask import Flask, request, jsonify , render_template #pip install flask
from werkzeug.utils import secure_filename
import fitz # pip install PyMuPDF
from tempfile import NamedTemporaryFile
from PIL import Image # pip install pillow
from tqdm import tqdm # pip install tqdm
import os
import cv2 # pip install opencv-python opencv-contrib-python
import numpy as np
import base64
import matplotlib.pyplot as plt # pip install matplotlib
import pytesseract
import re

from flask_cors import CORS, cross_origin

# for project , from project
import recognizers 
import extractors 
import compFunc
import box_detection
import TestFunc


def pix_to_numpy(pix):
    """Convert a fitz.Pixmap to a NumPy array."""
    if pix.n < 5:  # GRAY or RGB
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    else:  # CMYK
        pix = fitz.Pixmap(fitz.csRGB, pix)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    return img_array


def display_image(img_array, title):
    """Display an image for visual inspection."""
    plt.imshow(img_array)
    plt.title(title)
    plt.axis('off')
    plt.show()
 
def extract_images_from_page(doc):
    images = []

    for i in tqdm(range(len(doc)), desc="pages"):
        for img in tqdm(doc.get_page_images(i), desc="page_images"):
            xref = img[0]
            base_image = doc.extract_image(xref)
            pix = fitz.Pixmap(doc, xref)

            # Convert the Pixmap to a NumPy array
            img_array = pix_to_numpy(pix)

            print('Test using py')
            #box_detection.box_extraction(img_array, "~/Desktop/")
            #TestFunc.TableFromImg(img_array) # use may be regrettable ! Update needed

            image_text =''
            readable = recognizers.ImgReadable_assessment(img_array)
            if readable == True :
                #gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                th, threshed = cv2.threshold(gray,127,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                print('testing using different thresholding')
                print(th)
                #varA = compFunc.dframe(img_array)
                #extractors.display_image(threshed,'test-'+str(i))
                #extractors.display_image(gray,'test'+str(i))
                #test_text += Img2Text(gray) # less efficient
                details = Img2Text(img_array) # more efficient
                #textFilter(image_text)
                return details
            else:
                details = {'result':'N:Not readable'}
        if i==0:
            details = {'result':'N:Not readable'}
            return details
        else:
            details = {'result':'N:Not readable'}
            pass
    return details

def Img2Text(image):
    imageText = pytesseract.image_to_string(image,lang='eng')
    #print(imageText)
    
    detailsDict  = textFilter(imageText)
    imageText = 'Test'
    return detailsDict

def textFilter(text):
    final = []
    for word in text.split("\n"):
        if "”—" in word:
            word = word.replace("”—", ":")

        if "!" in word:
            nik_char = word.split()

        if "?" in word:
            word = word.replace("?", "7")

        final.append(word)

    #print(final)
    data_output = [
        [
            'Company Identification Number(CIN)'
            #'Name of Manufacturer/Vendor/OEM',
            #'Name of Contact Person',
            #'Office Address',
            #'City & PIN Code',
            #'Mobile No',
            #'E-Mail Id',
        ]
    ] 

    details = {
        'CIN': None,
        'Name': None,
        'Mobile Numbers': [],
        'Email IDs': [],
        'LandLine':[],
        'ICCID':[],
        'IMEI':[],
        'MISDN':[],
        'ManufacturerName':[],
        'StreetAddr':[],
        'CityAddr':[]
    }

    cin_pattern = re.compile(r"Company Identification Number \(CIN\)\*:\s*_?([A-Za-z0-9]+)")
    name_pattern = re.compile(r"Name of (?:Contact Person)\*:\s*(.*)")#|(
    manufacture_pattern = re.compile(r"Name of (?:Manufacturer/Vendor/OEM)\*:\s*(.*)")
    mobile_pattern = re.compile(r"\*([0-9]{10})")
    email_pattern = re.compile(r"\*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
    landline_pattern = re.compile(r'Landline No\*:\s*(011-?[0-9]{4}[-\s]?[0-9]{4})')
    officialAddr_patterm = re.compile(r'Office Address\*:\s*([A-Za-z0-9]+\s*-?(.*)\s*)+')
    cityAddr_pattern = re.compile(r'City & PIN Code\*:\s*([A-Za-z0-9]*\s*[0-9A-Za-z]+\s*-?(.*)\s*)+')

    # Define the keywords
    keywords = ["Vendor", "IMEI", "Device", "Vehicle", "E-SIM Provider", "ICCID", "Primary MISDN", "Fallback MISDN"]
    stopWords = ['Registration', 'Location', 'Tracking', 'Emergency', 'Alert', 'System', 'Registration', 'Ly', 'Modify', 'Information', 'Research', 'and', 'Development', 'Establishment', 'Details', 'for', 'One', 'Time', 'Testing', 'Process', 'with', 'backend']
    #tColHead_pattern = re.compile(r"\*:\s*_?([A-Za-z0-9]+)")

    #pattern = re.compile(r'[A-Z0-9]+[0-9]*[—]*\[.*?\]|[0-9]*[A-Z]*')
    vendor_pattern = re.compile(r'([A-Z\s]+)\s(\d+)')
    
    device_model_pattern = re.compile(r'[A-Za-z0-9\s]+(?=\s—)')
    
    iccid_pattern = re.compile(r'\b[1-9]\d{16,17}\b')
    imei_pattern = re.compile(r'\b[1-9]\d{12,15}\b')
    misdn_pattern = re.compile(r'\b[1-9]\d{12}\b')

    for line in final:
        # Extract CIN
        cin_match = cin_pattern.search(line)
        if cin_match:
            details['CIN'] = cin_match.group(1)
        
        # Extract Names
        name_match = name_pattern.search(line)
        if name_match:
            details['Name'] = name_match.group(1)
        
        manufacture_match = manufacture_pattern.findall(line)
        if manufacture_match:
                details['ManufacturerName'].append(manufacture_match)
        
        # Extract Mobile Numbers
        for mobile_match in mobile_pattern.findall(line):
            details['Mobile Numbers'].append(mobile_match)
        
        # Extract Email IDs
        for email_match in email_pattern.findall(line):
            details['Email IDs'].append(email_match)

        # Extract Email IDs
        for landline_match in landline_pattern.findall(line):
            details['LandLine'].append(landline_match)

        for officialAddr_match in officialAddr_patterm.findall(line):
            details['StreetAddr'].append(officialAddr_match)

        for cityAddr_match in cityAddr_pattern.findall(line):
            details['CityAddr'].append(cityAddr_match)


        # Convert the list of keywords to a set of words for faster checking
        keywords_set = set(word.lower() for keyword in keywords for word in keyword.split())
        results = []
        
        for imei_match in imei_pattern.findall(line):
            details['IMEI'].append(imei_match)
        for iccid_match in iccid_pattern.findall(line):
            details['ICCID'].append(iccid_match) 
        for misdn_match in misdn_pattern.findall(line):
            details['MISDN'].append(misdn_match)

    print('results')
    print(results)
    print('details')
    print(details)
    print('test')

    return details