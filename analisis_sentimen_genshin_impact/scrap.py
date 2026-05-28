import requests
from bs4 import BeautifulSoup

url = "https://pijak.dicoding.com/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

print(response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

# ambil title
print("TITLE:")
print(soup.title.text)

# ambil semua heading
print("\nHEADINGS:")
for h in soup.find_all(["h1", "h2", "h3"]):
    print(h.text.strip())

import pandas as pd

headings = []

for h in soup.find_all(["h1", "h2", "h3"]):
    headings.append(h.text.strip())

df = pd.DataFrame({
    "heading": headings
})

df.to_csv("hasil_scraping.csv", index=False)

print("Berhasil disimpan")