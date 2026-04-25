import re
import requests
from PDF import PdfReader

# Download the PDF
pdf_url = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
response = requests.get(pdf_url)

with open("budget.pdf", "wb") as f:
    f.write(response.content)

# Read the PDF
reader = PdfReader("budget.pdf")

pdf_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:  # Only add if text is extracted
        pdf_text += text

# Clean text: replace multiple spaces/newlines with a single space
pdf_text = re.sub(r'\s+', ' ', pdf_text)

print(pdf_text[:1000])  # Preview first 1000 characters
