import os
import re
import pandas as pd

# Create directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

print("Downloading and loading CSV dataset...")

# Load CSV dataset
url = "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
try:
    df = pd.read_csv(url)
    print(f"✓ CSV loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    exit(1)

# Clean dataset
print("Cleaning dataset...")
df = df.dropna()
df = df.drop_duplicates()
df.columns = df.columns.str.strip().str.lower()

print(f"✓ Dataset cleaned. Shape after cleaning: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst row sample:")
print(df.iloc[0])

# Convert DataFrame to formatted text that preserves all information
print("\nFormatting text...")
csv_text = ""
for idx, row in df.iterrows():
    if idx % 100 == 0:
        print(f"  Processing row {idx}...")
    # Create a nicely formatted row with all information preserved
    row_dict = row.to_dict()
    row_text = " | ".join([f"{col}: {value}" for col, value in row_dict.items()])
    csv_text += row_text + "\n"

# Only clean up extra whitespace, but preserve all alphanumeric and important characters
print("Normalizing whitespace...")
csv_text = re.sub(r'\s+', ' ', csv_text)

# Save cleaned text for chunking
print(f"Saving to file...")
with open("data/processed/full_text.txt", "w", encoding="utf-8") as f:
    f.write(csv_text)

print(f"\n✓ Cleaned text saved to data/processed/full_text.txt")
print(f"  Total text length: {len(csv_text)} characters")
print(f"  Sample of cleaned text (first 300 chars):")
print(f"  {csv_text[:300]}...")
