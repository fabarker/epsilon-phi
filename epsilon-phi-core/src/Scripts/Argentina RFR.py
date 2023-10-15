import pandas as pd
from docx import Document

# Path to the Word document
doc_path = "/Users/francisbarker/Desktop/Argentina Risk Free Rate.docx"

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

    pct = float(value.strip().replace('%', '').replace(',','')) / 100
    dateval = pd.to_datetime(date.strip())

    dates.extend([dateval])
    values.extend([pct])

df_ = pd.DataFrame(values, index=dates, columns=['Rate'])
# Display the dataframe
print(df.head())
