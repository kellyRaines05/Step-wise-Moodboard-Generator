import requests
from bs4 import BeautifulSoup

search_query = "automate search python"
cosmos_url = f"https://www.cosmos.so/search/elements/{search_query}"
pinterest_url = f"https://www.pinterest.com/search/pins/?q={search_query}&rs=typed"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

for g in soup.find_all('img', class_='g'):
    title_element = g.find('h3')
    if title_element:
        print(title_element.text)