
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from datetime import date
from backend.image_search import *
from backend.MBGen import MBGen
from backend.request_models import *

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

MBGEN = MBGen(user_prompt=None)

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

@app.post("/api/get_images")
def create_moodboard(input: GenerationRequest) -> KeywordsResponse:
    MBGEN.user_prompt = input.prompt
    if input.image_url:
        MBGEN.image = input.image_url
    keywords = MBGEN.generate_keywords()
    return KeywordsResponse(keywords=keywords.split(", "))

@app.post("/api/save_moodboard")
def save_moodboard(moodboard: Moodboard) -> SavedMoodboardResponse:
    filename = ""
    return SavedMoodboardResponse(status="success", filename=filename)

