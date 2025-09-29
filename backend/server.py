
from fastapi import FastAPI
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

mbgen = MBGen(user_prompt=None)

@app.get("/api/health")
def health_check() -> StatusResponse:
    return StatusResponse(status_code=200)

@app.get("/api/past_moodboards")
def get_past_moodboards() -> PastMoodboardsResponse:
    moodboards_dir = os.path.abspath('past_moodboards')
    print(f"Looking for moodboards in: {moodboards_dir}")
    moodboards = []
    if os.path.exists(moodboards_dir):
        for filename in os.listdir(moodboards_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(moodboards_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        past_moodboard = Moodboard(**data)
                        moodboards.append(past_moodboard)
                except Exception:
                    continue
    return PastMoodboardsResponse(moodboards=moodboards)

@app.post("/api/generate_title")
def title_moodboard(input: GenerationRequest) -> TitleResponse:
    mbgen.user_prompt = input.prompt
    generated_title = mbgen.generate_title()
    return TitleResponse(title=generated_title)

@app.post("/api/generate_moodboard")
def create_moodboard(prompt: GenerationRequest, title=None) -> Moodboard:
    mbgen.user_prompt = prompt.prompt
    keywords = mbgen.generate_keywords(image=None, user_prompt=prompt.prompt, title=title)
    
    return Moodboard(title=title, prompt=prompt.prompt, date_created=date.today(), image_url=None)

@app.post("/api/save_moodboard")
def save_moodboard(moodboard: Moodboard) -> SavedMoodboardResponse:
    filename = ""
    return SavedMoodboardResponse(status="success", filename=filename)

