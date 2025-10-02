
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import requests
import shutil
from backend.MBGen import MBGen
from backend.request_models import *
from backend.image_search import get_images
from backend.image_organization import ImageOrganization

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_DIR = os.path.abspath(os.path.join('past_moodboards', 'images'))
os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount("/past_moodboards", StaticFiles(directory=IMAGE_DIR), name="past_moodboards")

MOODBOARD_IMAGE_DIR = os.path.abspath('images')

MBGEN = MBGen(user_prompt=None)
ORGANIZER = ImageOrganization(title=None)

@app.get("/api/health")
def health_check() -> StatusResponse:
    return StatusResponse(status_code=200)

@app.get("/api/get_past_moodboards")
def get_past_moodboards() -> PastMoodboardsResponse:
    past_moodboards = os.path.abspath(os.path.join('past_moodboards', 'past_moodboards.json'))
    moodboards = []
    if os.path.exists(past_moodboards):
        with open(past_moodboards, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                past_moodboard = Moodboard(**item)
                moodboards.append(past_moodboard)
    return PastMoodboardsResponse(moodboards=moodboards)

@app.post("/api/generate_title")
def title_moodboard(input: GenerationRequest) -> TitleResponse:
    MBGEN.user_prompt = input.prompt
    generated_title = MBGEN.generate_title()
    return TitleResponse(title=generated_title)

@app.post("/api/get_keywords")
def get_moodboard_images(input: GenerationRequest) -> KeywordsResponse:
    MBGEN.user_prompt = input.prompt
    if input.image_url:
        MBGEN.image = input.image_url
    keywords = MBGEN.generate_keywords()

    return json.loads(keywords)

@app.post("/api/get_images")
def get_moodboard_iamges(keyword: ImageRequest) -> ImagesResponse:
    images = get_images(keyword.keyword)
    return ImagesResponse(images=images, keyword=keyword.keyword)

@app.post("/api/download_images")
def download_images(images: DownloadRequest) -> StatusResponse:
    for i, image_url in enumerate(images.images):
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            with open(os.path.join(MOODBOARD_IMAGE_DIR, f"image_{i}.png"), "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return StatusResponse(status_code=200)
        except requests.exceptions.RequestException as e:
            return StatusResponse(status_code=500, detail=e)
        except Exception as e:
            return StatusResponse(status_code=500, detail=e)

@app.post("/api/organize_moodboard")
def organize_moodboard(title: TitleResponse):
    ORGANIZER.title = title.title
    ORGANIZER.organize_images()
    return

@app.post("/api/save_moodboard")
def save_moodboard() -> SavedMoodboardResponse:
    json_moodboard = ORGANIZER.save_moodboard()
    for filename in os.listdir(MOODBOARD_IMAGE_DIR):
        file_path = os.path.join(MOODBOARD_IMAGE_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return SavedMoodboardResponse(status="success", filename=json_moodboard)

