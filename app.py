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

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET'])
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
                        if is_text_page(page):
                            textOrImg = str(is_text_page(page))
                        elif is_image_page(page) : 
                            textOrImg = str(is_image_page(page))
                            print('calling funcLaplacian')
                            extract_images_from_page(doc)
                            #funcLaplacian(page)
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
                return jsonify({'text': text})
            except Exception as e:
                print(f"Error extracting text: {str(e)}")
                return jsonify({'error': 'Error processing PDF'}), 500

        else:
            return jsonify({'error': 'Invalid file format. Please upload a PDF.'}), 400

    return jsonify({'error': 'Method not allowed'}), 405

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



def extract_images_from_page(doc):
    images = []

    for i in tqdm(range(len(doc)), desc="pages"):
        for img in tqdm(doc.get_page_images(i), desc="page_images"):
            xref = img[0]
            base_image = doc.extract_image(xref)
            pix = fitz.Pixmap(doc, xref)
            
            # Convert the Pixmap to a NumPy array
            img_array = pix_to_numpy(pix)
            
            # Ensure the image is in RGB format
            if img_array.shape[2] == 4:  # If RGBA, convert to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif img_array.shape[2] == 1:  # If grayscale, convert to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

            img_array2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            display_image(img_array2,'gray')
            display_image(img_array,'rgb')
            # Calculate the Laplacian variance
            laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
            print(f'Laplacian variance: {laplacian_var}')

            # Save the image as a base64 string
            image_base64 = base64.b64encode(base_image['image']).decode('utf-8')
            images.append({
                'image_index': xref,
                'image_base64': image_base64,
                'image_format': base_image['ext'],
                'laplacian_var': laplacian_var
            })

    return images

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
    plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    app.run(debug=True)