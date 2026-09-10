from PIL import Image, ImageFilter, ImageFile
import pytesseract
import pandas as pd
import re
import cv2
from PIL import Image
import datetime
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
path = r'C:\Program Files\Tesseract-OCR'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    img_gray = img.convert('L')

    # Convert image to numpy array and threshold to get a binary image
    np_img = np.array(img_gray)
    _, binary_img = cv2.threshold(np_img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert binary image back to PIL format for further processing
    pil_img = Image.fromarray(binary_img)

    # Optionally apply a median filter for denoising
    denoised_img = pil_img.filter(ImageFilter.MedianFilter(size=3))

    # OCR using pytesseract
    text = pytesseract.image_to_string(pil_img)
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
    lines = text.split('\n')

    df = pd.DataFrame()
    for line in lines:
        splt = [re.sub(r'\s+', '', x).replace('_','') for x in line.split('|')]
        df_row = pd.DataFrame(splt).T
        df = pd.concat((df, df_row), axis=0)


    nwlines = '|'.join(stripped).replace('||','|').replace('--','-').split('|')

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

    data_path = r'/Users/francisbarker/Desktop/GFD EM Equity/GFD EM Equity.xlsx'
    df = pd.read_excel(data_path, sheet_name=None, index_col=0)

    data = pd.DataFrame()
    for region in df.keys():
        df_r = df.get(region)

        df_region = pd.DataFrame()
        for year, row in df_r.iterrows():
            row_nans = row.dropna()
            row_nans.index = [datetime.date(year=int(year), month=int(x), day=1) for x in row_nans.index]
            df_region = pd.concat((df_region, row_nans), axis=0)
        df_region.columns = [region]
        df_region = df_region[~df_region.index.duplicated(keep='first')]
        df_region.index = pd.to_datetime(df_region.index)
        data = pd.concat((data, df_region), axis=1)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data.to_clipboard()

