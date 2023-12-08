from pdf2image import convert_from_path
from PIL import Image
import sys, os
import pytesseract
# import easyocr

# Run this script with the following command:
# python3 ocr.py ./data/file.pdf

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
# reader = easyocr.Reader(['en']) # need to run only once to load model into memory

# add path argument
pdf_path = str(sys.argv)

# convert pdf to images
convert_from_path(pdf_path, output_folder='./data/images', fmt='png', output_file='output')
# open the txt file
with open('./data/output.txt', 'a') as f:
    # loop the files inside the output_folder
    for image in os.listdir('./data/images'):
        # get absolute path of the image
        img = Image.open(os.path.join('./data/images', image))
        # convert the image to text
        text = pytesseract.image_to_string(img)
        # write the result to the txt file
        f.write(text)
        # print the text
        print(text)
    # close the txt file
    f.close()

