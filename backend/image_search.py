import os
from pexels_api import API

def get_images(query: str):
    api = API(os.getenv("PEXELS_API"))

    page = 1
    results_per_page = 3
    api.search(query, page=page, results_per_page=results_per_page)

    photos = api.get_entries()

    image_urls = [photo.original for photo in photos]
    return image_urls