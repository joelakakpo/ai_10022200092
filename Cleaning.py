import os
import re
import pandas as pd

# Create directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Load CSV dataset
url = "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
df = pd.read_csv(url)

# Clean dataset
df = df.dropna()
df = df.drop_duplicates()
df.columns = df.columns.str.strip().str.lower()

print(df.head())

# Convert DataFrame to text
csv_text = ""
for _, row in df.iterrows():
    row_text = " ".join([str(value) for value in row])
    csv_text += row_text + "\n"

# Clean numbers only from csv_text (not pdf_text here!)
csv_text = re.sub(r'[^0-9. ]', '', csv_text)

# Save cleaned text for chunking
with open("data/processed/full_text.txt", "w", encoding="utf-8") as f:
    f.write(csv_text)

print("Text saved to data/processed/full_text.txt")