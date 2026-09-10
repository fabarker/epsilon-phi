import pandas as pd
from docx import Document

# Path to the Word document
doc_path = "/Users/francisbarker/Desktop/Jordan Risk Free Rate.docx"

# Load the Word document with python-docx
doc = Document(doc_path)

# Initialize a list to hold our document data
data = []

# Loop through each paragraph in the Word document
for para in doc.paragraphs:
    # Check if paragraph is not empty
    if para.text.strip() != "":
        # Split the paragraph text by '|' and append it to data
        data.append(para.text.split('|'))

# Transpose data since we want to split text into columns
df = pd.DataFrame(data).T


# Name the column 'data'
df.columns = ['data']


dates = list()
values = list()
for idx, row in df.iterrows():
    date, value = row.get('data').split('=')

    pct = float(value.strip().replace('%', '').replace(',',''))
    dateval = pd.to_datetime(date.strip())

    dates.extend([dateval])
    values.extend([pct])

df_ = pd.DataFrame(values, index=dates, columns=['Rate'])

SD = df_.index.min()
ED = df_.index.max()
all_dates = pd.date_range(SD, ED)
df_dates = df_.reindex(all_dates).ffill().reindex(pd.date_range(SD, ED, freq='B'))

print(df_dates.head())
