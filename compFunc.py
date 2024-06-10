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
import box_detection


def dframe(imgData):
    img = imgData.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    print('Test dframe')
    minLineLength =2
    maxLineGap = 4
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 350, minLineLength, maxLineGap, 10)
    print('done with lines')
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("image", edges)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    return '1'

'''
from img2table.document import PDF
from img2table.ocr import TesseractOCR

# Instantiation of the pdf
pdf = PDF(src="/home/rd/Downloads/SignedRegForm (16).pdf")

# Instantiation of the OCR, Tesseract, which requires prior installation
ocr = TesseractOCR(lang="eng")

# Table identification and extraction
pdf_tables = pdf.extract_tables(ocr=ocr)

# We can also create an excel file with the tables
pdf.to_xlsx('/home/rd/Downloads/tables.xlsx',
            ocr=ocr)'''

'''

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
from imutils import contours as cont
from collections import defaultdict

from flask_cors import CORS, cross_origin

# for project , from project
import recognizers 
import extractors 
import compFunc

def getTableValue(table, img, img_ocr, fsize):
    #img_ocr = img.copy()
    #img_ocr = cv2.cvtColor(img_ocr,cv2.COLOR_BGR2GRAY)
    data = []
    header = []
    for i,row in enumerate(table):
        data_row = []
        for cell in row:
            crop = img_ocr[cell[1]+2:cell[1]+cell[3]-2, cell[0]+2:cell[0]+cell[2]-2]
            #cv2.imwrite(str(i)+".png",crop)
            cell_text = getTextOfBox(crop)
            if i == 0:
                header.append(cell_text)
                cv2.rectangle(img, (cell[0], cell[1]), (cell[0] + cell[2], cell[1] + cell[3]), (0,255,0), -1)
            else:
                cv2.rectangle(img, (cell[0], cell[1]), (cell[0] + cell[2], cell[1] + cell[3]), (0,255,255), -1)
                data_row.append(cell_text)
            img = putTextUTF8(img, cell_text, (cell[0],cell[1]), fsize)
        if i == 0:
            data.append(header)
        else:
            data.append(data_row)
    return data, img

def findTable(arr):
    table = defaultdict(list)
    for i,b in enumerate(arr):
        if b[2] < b[3]/2:
            continue
        table[str(b[1])].append(b)
    #print(table)
    table = [i[1] for i in table.items()]# if len(i[1]) > 1]
    #print(([len(x) for x in table]))
    num_cols = max([len(x) for x in table])
    #print("num_cols:",num_cols)
    table = [i for i in table if len(i) == num_cols]
    #print("table rows=", len(table))
    #print("table cols=",num_cols)
    print("table size:{}x{}".format(len(table), num_cols))
    return table

def TableFromImg(img):
    table = getTable(img, y_start=0, min_w=10, min_h=10)
    img2 = img.copy()
    data,img = getTableValue(table, img, img2, 10)
    print(data)
    cv2.imshow('img-last',img)
    print('done-1')
    return
def getTextOfBox(img):
    return pytesseract.image_to_string(img).strip()#.lower()

def findMinMaxRow(v_img):
    aleft, aright = 0, 0
    list_col = []
    w, h = v_img.shape[0], v_img.shape[1]
    for r in range(w-1):
        pixel_white = 0
        for c in range(h-1):
            if v_img[r,c] == 255:
                pixel_white += 1
        if pixel_white > 20:
            list_col.append(r)
    aleft, aright = min(list_col), max(list_col)
    return aleft, aright

def reDrawLine(img, aleft, aright, same_len=True):
    w, h = img.shape[0], img.shape[1]
    for r in range(w-1):
        pixel_white = 0
        start = 0
        end = 0
        for c in range(h-1):
            if img[r,c] == 255:
                pixel_white += 1
            if img[r, c] == 0 and img[r,c+1] == 255:
                start = c
            if img[r, c] == 255 and img[r,c+1] == 0:
                end = c
        if pixel_white > 20:
            if same_len:
                img[r,aleft:aright] = 255
            else:
                img[r,start:end] = 255
    return img


def getTable(src_img, y_start=0, min_w=10, min_h=10):
    if y_start != 0:
        src_img = src_img[y_start:,:]
    if len(src_img.shape) == 2:
        gray_img = src_img
    elif len(src_img.shape) ==3:
        gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -3)
    h_img = thresh_img.copy()
    v_img = thresh_img.copy()
    scale = 15

    h_size = int(h_img.shape[1]/scale)
    h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(h_size,1))

    h_erode_img = cv2.erode(h_img,h_structure,1)
    h_dilate_img = cv2.dilate(h_erode_img,h_structure,1)

    v_size = int(v_img.shape[0] / scale)
    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    v_erode_img = cv2.erode(v_img, v_structure, 1)
    v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

    
    aleft, aright = findMinMaxRow(v_dilate_img.T)
    aleft2, aright2 = findMinMaxRow(h_dilate_img)

    h_dilate_img = reDrawLine(h_dilate_img, aleft, aright, True)
    #v_dilate_img = reDrawLine(v_dilate_img.T, aleft, aright, False).T
    #cv2.imshow('h_dilate_img',h_dilate_img)
    #cv2.imshow('h_dilate_img',v_dilate_img)
    #cv2.waitKey()
    #list_hlines = getLines(h_dilate_img)
    #list_vlines = getLines(v_dilate_img.T)
    #print(len(list_hlines))
    #print(len(list_vlines))
    #for i,_ in list_hlines:
    #    for j,_ in list_hlines
    #exit()
    #v_dilate_img = reDrawLine(v_dilate_img.T, aleft2, aright2, True).T
    v_dilate_img.T[aleft,aleft2:aright2] = 255
    v_dilate_img.T[aright,aleft2:aright2] = 255
    
    edges = cv2.Canny(h_dilate_img,50,150,apertureSize = 3) 
    #print(len(edges))

    # This returns an array of r and theta values 
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200) 
    #print(len(lines))
    #cv2.waitKey()
    mask_img = h_dilate_img + v_dilate_img
    joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)
    #mask_img = 255 - mask_img
    #mask_img = unsharp_mask(mask_img)
    convolution_kernel = np.array(
                                [[0, 1, 0], 
                                [1, 2, 1], 
                                [0, 1, 0]]
                                )

    #mask_img = cv2.filter2D(mask_img, -1, convolution_kernel)
    #mask_img = 255- mask_img
    #cv2.imshow('mask', mask_img)
    #cv2.imshow('joints_img', joints_img)
    #cv2.waitKey()
    # cv2.imshow('join', joints_img)
    # cv2.waitKey()
    # fig, ax = plt.subplots(2,2)
    # fig.suptitle("table detect")
    # ax[0,0].imshow(h_dilate_img)
    # ax[0,1].imshow(v_dilate_img)
    # ax[1,0].imshow(mask_img)
    # ax[1,1].imshow(joints_img)
    # plt.show()cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    (contours, boundingBoxes) = cont.sort_contours(contours, method="left-to-right")
    (contours, boundingBoxes) = cont.sort_contours(contours, method="top-to-bottom")

    table = findTable([cv2.boundingRect(x) for x in contours])
    
    # for r in table:
    #     for c in r:

    #         cv2.rectangle(src_img,(c[0], c[1]),(c[0] + c[2], c[1] + c[3]),(0, 0, 255), 1)
    #         cv2.putText(src_img, , (c[0] + c[2]//2,c[1] + c[3]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 2)
    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     if (w >= min_w and h >= min_h):
    #         #count += 1
    #         if count != 0:
    #             cv2.rectangle(src_img,(x, y),(x + w, y + h),(0, 0, 255), 1)
    #             list_cells.append([x,y,w,h])
    #             cv2.putText(src_img, str(count), (x+w//2,y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    #         count += 1
    #cv2.waitKey()
    #cv2.imwrite('a.jpg', src_img)
    return table#mask_img, joints_img

def TableFromImg(img):
    #plt.imshow(img)
    #plt.show()
    # for adding border to an image
    img1= cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_CONSTANT,value=[255,255])

    img123 = img1.copy()
    
    # Thresholding the image
    thresh, th3 = cv2.threshold(img1, 127, 255,cv2.THRESH_BINARY )#| cv2.THRESH_OTSU
    
    #imgplot = plt.imshow(cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    # to flip image pixel values
    th3 = 255-th3
    
    #imgplot = plt.imshow(cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    # initialize kernels for table boundaries detections
    if(th3.shape[0]<1000):
        ver = np.array([[1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1]])
        hor = np.array([[1,1,1,1,1,1]])
        
    else:
        ver = np.array([[1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1]])
        hor = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    
    # to detect vertical lines of table borders
    img_temp1 = cv2.erode(th3, ver, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, ver, iterations=3)

    #imgplot = plt.imshow(cv2.resize(verticle_lines_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    # to detect horizontal lines of table borders
    img_hor = cv2.erode(th3, hor, iterations=3)
    hor_lines_img = cv2.dilate(img_hor, hor, iterations=4)

    #imgplot = plt.imshow(cv2.resize(hor_lines_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    # adding horizontal and vertical lines
    hor_ver = cv2.add(hor_lines_img,verticle_lines_img)

    #imgplot = plt.imshow(cv2.resize(hor_ver, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    hor_ver = 255-hor_ver

    #imgplot = plt.imshow(cv2.resize(hor_ver, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    # subtracting table borders from image
    temp = cv2.subtract(th3,hor_ver)

    #imgplot = plt.imshow(cv2.resize(temp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    temp = 255-temp

    #imgplot = plt.imshow(cv2.resize(temp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    #Doing xor operation for erasing table boundaries
    tt = cv2.bitwise_xor(img1,temp)
    
    #imgplot = plt.imshow(cv2.resize(tt, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    iii = cv2.bitwise_not(tt)

    #imgplot = plt.imshow(cv2.resize(iii, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    tt1=iii.copy()
    
    #imgplot = plt.imshow(cv2.resize(tt1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()

    #kernel initialization
    ver1 = np.array([[1,1],
                   [1,1],
                   [1,1],
                   [1,1],
                   [1,1],
                   [1,1],
                   [1,1],
                   [1,1],
                   [1,1]], dtype=np.uint8)
    
    hor1 = np.array([[1,1,1,1,1,1,1,1,1,1],
                   [1,1,1,1,1,1,1,1,1,1]], dtype=np.uint8)
    
    #morphological operation
    temp1 = cv2.erode(tt1, ver1, iterations=1)
    verticle_lines_img1 = cv2.dilate(temp1, ver1, iterations=1)
    
    #imgplot = plt.imshow(cv2.resize(verticle_lines_img1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()
    
    temp12 = cv2.erode(tt1, hor1, iterations=1)
    hor_lines_img2 = cv2.dilate(temp12, hor1, iterations=1)
    
    #imgplot = plt.imshow(cv2.resize(hor_lines_img2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()
    
    # doing or operation for detecting only text part and removing rest all
    hor_ver = cv2.add(hor_lines_img2,verticle_lines_img1)
    
    #imgplot = plt.imshow(cv2.resize(hor_ver, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()
    
    dim1 = (hor_ver.shape[1],hor_ver.shape[0])

    dim = (hor_ver.shape[1]*2,hor_ver.shape[0]*2)
    
    # resizing image to its double size to increase the text size
    resized = cv2.resize(hor_ver, dim, interpolation = cv2.INTER_AREA)
    
    #bitwise not operation for fliping the pixel values so as to apply morphological operation such as dilation and erode
    want = cv2.bitwise_not(resized)
    
    #imgplot = plt.imshow(cv2.resize(want, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    #plt.show()
    
    if(want.shape[0]<1000):
        kernel1 = np.array([[1,1,1]], dtype=np.uint8)
        kernel2 = np.array([[1,1,1],
                            [1,1,1]], dtype=np.uint8)
        kernel3 = np.array([[1,0,1],[0,1,0],
                           [1,0,1]], dtype=np.uint8)
    else:
        kernel1 = np.array([[1,1,1,1,1]], dtype=np.uint8)
        kernel2 = np.array([[1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1]], dtype=np.uint8)

    tt1 = cv2.dilate(want,kernel1,iterations=14)

    imgplot = plt.imshow(cv2.resize(tt1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    plt.show()

    # getting image back to its original size
    resized1 = cv2.resize(tt1, dim1, interpolation = cv2.INTER_AREA)

    print('testing-1----194')

    # Find contours for image, which will detect all the boxes
    im21, contours1, hierarchy1 = cv2.findContours(resized1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('testing-2----198')
    #sim21, contours1, hierarchy1 = cv2.findContours(gray2, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    #print( cv2.findContours(threshed2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))

    #sorting contours by calling fuction
    (cnts, boundingBoxes) = sort_contours(contours1, method="top-to-bottom")
    
    #storing value of all bouding box height
    heightlist=[]
    for i in range(len(boundingBoxes)):
        heightlist.append(boundingBoxes[i][3])
    
    #sorting height values
    heightlist.sort()
    
    sportion = int(.5*len(heightlist))
    
    eportion = int(0.05*len(heightlist))
    
    #taking 50% to 95% values of heights and calculate their mean 
    #this will neglect small bounding box which are basically noise 
    try:
        medianheight = statistics.mean(heightlist[-sportion:-eportion])
    except:
        medianheight = statistics.mean(heightlist[-sportion:-2])
    
    #keeping bounding box which are having height more then 70% of the mean height and deleting all those value where 
    # ratio of width to height is less then 0.9
    box =[]
    imag = iii.copy()
    for i in range(len(cnts)):    
        cnt = cnts[i]
        x,y,w,h = cv2.boundingRect(cnt)
        if(h>=.7*medianheight and w/h > 0.9):
            image = cv2.rectangle(imag,(x+4,y-2),(x+w-5,y+h),(0,255,0),1)
            box.append([x,y,w,h])
        # to show image
    
    imgplot = plt.imshow(cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    plt.show()
    
    cv2.imshow('imagegen.jpg',image)

    #rearranging all the bounding boxes horizontal wise where every box fall on same horizontal line 
    main=[]
    j=0
    l=[]
    for i in range(len(box)):    
        if(i==0):
            l.append(box[i])
            last=box[i]
        else:
            if(box[i][1]<=last[1]+medianheight/2):
                l.append(box[i])
                last=box[i]
                if(i==len(box)-1):
                    main.append(l)
            else:
    #             print(l)            
                main.append(l)
                l=[]
                last = box[i]
                l.append(box[i])
    
    #calculating maximum number of box in a particular row
    maxsize=0
    for i in range(len(main)):
        l=len(main[i])
        if(maxsize<=l):
            maxsize=l   
    
    ylist=[]
    for i in range(len(boundingBoxes)):
        ylist.append(boundingBoxes[i][0])
    
    ymax = max(ylist)
    ymin = min(ylist)
    
    ymaxwidth=0
    for i in range(len(boundingBoxes)):
        if(boundingBoxes[i][0]==ymax):
            ymaxwidth=boundingBoxes[i][2]
    
    TotWidth = ymax+ymaxwidth-ymin
    
    width = []
    widthsum=0
    for i in range(len(main)):
        for j in range(len(main[i])):
            widthsum = main[i][j][2]+widthsum
        
    #     print(" Row ",i,"total width",widthsum)
        width.append(widthsum)
        widthsum=0
        
    
    #removing all the lines which are not the part of the table
    main1=[]
    flag=0
    for i in range(len(main)):
        if(i==0):
            if(width[i]>=(.8*TotWidth) and len(main[i])==1 or width[i]>=(.8*TotWidth) and width[i+1]>=(.8*TotWidth) or len(main[i])==1):
                flag = 1
        else:
            if(len(main[i])==1 and width[i-1]>=.8*TotWidth):
                flag=1
               
            elif(width[i]>=(.8*TotWidth) and len(main[i])==1):
                 flag=1
                 
            elif(len(main[i-1])==1 and len(main[i])==1 and (width[i]>=(.7*TotWidth) or width[i-1]>=(.8*TotWidth))):
                flag=1
        
            
        if(flag==1):
            pass
        else:
            main1.append(main[i])
        
        flag=0
    
    maxsize1=0
    for i in range(len(main1)):
        l=len(main1[i])
        if(maxsize1<=l):
            maxsize1=l  
    
    #calculating the values of the mid points of the columns 
    midpoint=[]
    for i in range(len(main1)):
        if(len(main1[i])==maxsize1):
    #         print(main1[i])
            for j in range(maxsize1):
                midpoint.append(int(main1[i][j][0]+main1[i][j][2]/2))
            break
    
    midpoint=np.array(midpoint)
    midpoint.sort()
    
    final = [[]*maxsize1]*len(main1)
    
    #sorting the boxes left to right
    for i in range(len(main1)):
        for j in range(len(main1[i])):
            min_idx = j        
            for k in range(j+1,len(main1[i])):
                if(main1[i][min_idx][0]>main1[i][k][0]):
                    min_idx = k
            
            main1[i][j], main1[i][min_idx] = main1[i][min_idx],main1[i][j]
    
    #storing the boxes in their respective columns based upon their distances from mid points  
    finallist = []
    for i in range(len(main1)):
        lis=[ [] for k in range(maxsize1)]
        for j in range(len(main1[i])):
    #         diff=np.zeros[maxsize]
            diff = abs(midpoint-(main1[i][j][0]+main1[i][j][2]/4))
            minvalue = min(diff)
            ind = list(diff).index(minvalue)
    #         print(minvalue)
            lis[ind].append(main1[i][j])
    #     print('----------------------------------------------')
        finallist.append(lis)
    
            
            
    
    #extration of the text from the box using pytesseract and storing the values in their respective row and column
    todump=[]
    for i in range(len(finallist)):
        for j in range(len(finallist[i])):
            to_out=''
            if(len(finallist[i][j])==0):
                print('-')
                todump.append(' ')
            
            else:
                for k in range(len(finallist[i][j])):                
                    y,x,w,h = finallist[i][j][k][0],finallist[i][j][k][1],finallist[i][j][k][2],finallist[i][j][k][3]
    
                    roi = iii[x:x+h, y+2:y+w]
                    roi1= cv2.copyMakeBorder(roi,5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255])
                    img = cv2.resize(roi1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    kernel = np.ones((2, 1), np.uint8)
                    img = cv2.dilate(img, kernel, iterations=1)
                    img = cv2.erode(img, kernel, iterations=2)
                    img = cv2.dilate(img, kernel, iterations=1)
                    
                    
    
                    out = pytesseract.image_to_string(img)
                    if(len(out)==0):
                        out = pytesseract.image_to_string(img,config='-psm 10')
                    
                    to_out = to_out +" "+out
                    
                print(to_out)
                    
                todump.append(to_out)
                cv2.imshow('image',img)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
        
    
                
                
               
        print("--------------------------------------------------")
        
        
    


#function to sort contours by its x-axis (top to bottom)
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
 
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
 
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
 
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
 
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


new code segment


from img2table.document import PDF
from img2table.ocr import TesseractOCR

# Instantiation of the pdf
pdf = PDF(src="/home/rd/Downloads/SignedRegForm (16).pdf")

# Instantiation of the OCR, Tesseract, which requires prior installation
ocr = TesseractOCR(lang="eng")

# Table identification and extraction
pdf_tables = pdf.extract_tables(ocr=ocr)

# We can also create an excel file with the tables
pdf.to_xlsx('/home/rd/Downloads/tables.xlsx',
            ocr=ocr)
*************************
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

def TableFromImg(image):
    BLUR_KERNEL_SIZE = (17, 17)
    STD_DEV_X_DIRECTION = 0
    STD_DEV_Y_DIRECTION = 0
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, STD_DEV_X_DIRECTION, STD_DEV_Y_DIRECTION)
    MAX_COLOR_VAL = 255
    BLOCK_SIZE = 15
    SUBTRACT_FROM_MEAN = -2
    
    gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    img_bin = cv2.adaptiveThreshold(
        gray_image,
        MAX_COLOR_VAL,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        SUBTRACT_FROM_MEAN,
    )
    vertical = horizontal = img_bin.copy()
    SCALE = 5
    image_width, image_height = horizontal.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image_width / SCALE), 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image_height / SCALE)))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)
    
    horizontally_dilated = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vertically_dilated = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    
    mask = horizontally_dilated + vertically_dilated
    contours, heirarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    MIN_TABLE_AREA = 1e5
    contours = [c for c in contours if cv2.contourArea(c) > MIN_TABLE_AREA]
    perimeter_lengths = [cv2.arcLength(c, True) for c in contours]
    epsilons = [0.1 * p for p in perimeter_lengths]
    approx_polys = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, epsilons)]
    bounding_rects = [cv2.boundingRect(a) for a in approx_polys]

    # The link where a lot of this code was borrowed from recommends an
    # additional step to check the number of "joints" inside this bounding rectangle.
    # A table should have a lot of intersections. We might have a rectangular image
    # here though which would only have 4 intersections, 1 at each corner.
    # Leaving that step as a future TODO if it is ever necessary.
    images = [image[y:y+h, x:x+w] for x, y, w, h in bounding_rects]
    cv2.imshow('testing new func',image)
    return images



    '''