import cv2
import pytesseract

path = r'C:\ProgramFiles\Tesseract - OCR'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread(r'C:\Users\fabar\Documents\cashflows.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Apply OCR using Tesseract
data = pytesseract.image_to_string(thresh, config='--psm 6')

# Process the extracted data
lines = data.split('\n')
matrix = []
for line in lines:
    row = line.split()
    matrix.append(row)

