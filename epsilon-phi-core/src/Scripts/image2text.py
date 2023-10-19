from PIL import Image, ImageFile
import pytesseract
import pandas as pd
import re
import cv2
import datetime
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
path = r'C:\ProgramFiles\Tesseract - OCR'
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(img)
    return text


def is_year(string):
    """
    Check if the given string represents a valid year.

    Args:
    - string (str): The string to check.

    Returns:
    - bool: True if the string represents a year, False otherwise.
    """
    # Ensure the string has a length of 4 and all characters are digits
    if len(string) == 4 and string.isdigit() and '.' not in string:
        year_value = int(string)
        if 1700 <= year_value <= datetime.date.today().year:
            return True
    return False


def text_to_dataframe(text):
    lines = np.array(text.replace('\n\n', '\n').replace("|", "").split('\n'))
    bl = np.array([is_year(x) for x in lines])

    data = []

    # Regular expression to identify years
    year_pattern = re.compile(r'(\d{4})')

    for line in lines:
        match = year_pattern.search(line)
        if match:
            year = match.group(1)
            # Split line by spaces to get data, ignore the year column
            values = line.split()[1:]
            data.append([year] + values)

    # Assuming the first line contains months (1-12)
    columns = ['Year'] + lines[0].split()[1:]

    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    image_path = "/Users/francisbarker/Desktop/GFD 10 Year Yields/GBP/IMG_3484.jpg"
    extracted_text = extract_text_from_image(image_path)
    df = text_to_dataframe(extracted_text)
    print(df)

