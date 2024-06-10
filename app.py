# leveraging routes as API 
# to ensure usability of the project 
# by using just simple curl or ajax call

from flask import Flask, request, jsonify , render_template #pip install flask
from werkzeug.utils import secure_filename
import fitz # pip install PyMuPDF
from tempfile import NamedTemporaryFile
from PIL import Image # pip install pillow
from tqdm import tqdm # pip install tqdm
import os
import cv2 # pip install opencv-python opencv-contrib-python
import numpy as np # pip install numpy
import base64 # iff not found - pip install pybase64
import matplotlib.pyplot as plt # pip install matplotlib

# for integration
from flask_cors import CORS, cross_origin # pip install flask-cors

# for project , from project
import recognizers 
import extractors 


app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET'])
@cross_origin()#ip here with port
def home():
    return render_template("main.html")

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'No PDF file uploaded'}), 400

        pdf_file = request.files['pdf_file']
        cert_num = request.form.get('certNum')

        if pdf_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if pdf_file and allowed_file(pdf_file.filename):
            filename = secure_filename(pdf_file.filename)

            try:
                with NamedTemporaryFile(delete=False) as temp_file: # test with temporary file 
                    temp_file.write(pdf_file.read())
                    doc = fitz.open(temp_file.name)  # Use temporary file path
                    text = "Not Found"
                    textOrImg = ""
                    images=[]
                    found = False
                    for page in doc:
                        if recognizers.is_text_page(page):
                            print('calling extracting sequence from textual pdf')
                            textOrImg = 'textual pdf'
                        elif recognizers.is_image_page(page) : 
                            #textOrImg = str(is_image_page(page))
                            textOrImg = 'image pdf'
                            print('calling extracting sequence from image')
                            details = extractors.extract_images_from_page(doc)
                            #funcLaplacian(page)
                            print('back')
                            break
                        else : 
                            textOrImg = 'nAn'
                        #  = ?is_text_page:
                        
                        text += page.get_text("text")# + "\n"  # Extract text with newline
                        #text = page.get_text("text") + "\n"  # Extract text with newline page specific
                        if cert_num.lower() in text.lower():  # Case-insensitive search
                        #if cert_num in ''.join(text.split()).lower():
                            found = True
                            text = 'found-1'+cert_num
                            break
                        else:
                            found = False
                            text = 'Not-Found'+cert_num
                        # else:
                            # if(text.find(cert_num.lower())):
                                # found = True
                                # text = 'found-2'
                                # break
                            # else:
                                # text = "not found"

                    doc.close()
                    # Optionally remove temporary file after processing
                    temp_file.close()  # Close the file before deleting
                text += '--' + textOrImg 
                return jsonify({'text': text},{'details':details})
            except Exception as e:
                print(f"Error extracting text: {str(e)}")
                return jsonify({'error': 'Error processing PDF'}), 500

        else:
            return jsonify({'error': 'Invalid file format. Please upload a PDF.'}), 400

    return jsonify({'error': 'Method not allowed'}), 405

# driver
if __name__ == '__main__':
    app.run(debug=True)