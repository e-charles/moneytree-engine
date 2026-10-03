from pathlib import Path
from bs4 import BeautifulSoup
import requests

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

result = session.get('https://etfdb.com/etfs/sector/consumer-discretionaries/')
Path("response.html").write_text(result.text, encoding="utf-8")

soup = BeautifulSoup(result.text, "html.parser")

print("table rows:", len(soup.select("table tbody tr")))
print("script tags:", len(soup.find_all("script")))
print("contains target text:", "Example Fund Name" in result.html)