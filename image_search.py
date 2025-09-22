import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib.parse
from io import BytesIO
from PIL import Image

def get_images(keywords: str) -> list:
    images_results = []
    driver = webdriver.Chrome()

    try:
        search_query = urllib.parse.quote(keywords)
        cosmos_url = f"https://www.cosmos.so/search/elements/{search_query}"
        
        driver.get(cosmos_url)

        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "img")))
        image_elements = driver.find_elements(By.TAG_NAME, "img")
        
        for img in image_elements:
            img_src = img.get_attribute('src')
            if img_src and img_src.startswith('http'):
                try:
                    response = requests.get(img_src, timeout=10)
                    image_data = response.content
                    image_stream = BytesIO(image_data)
                    image_object = Image.open(image_stream)
                    images_results.append(image_object)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to fetch image data from {img_src}: {e}")
                except Exception as e:
                    print(f"Failed to process image from {img_src}: {e}")
    finally:
        driver.quit()
    
    return images_results