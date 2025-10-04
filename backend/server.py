
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import requests
import shutil
import uuid
from datetime import date, datetime
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
app.mount("/past_moodboards", StaticFiles(directory=os.path.abspath('past_moodboards')), name="past_moodboards")

MOODBOARD_IMAGE_DIR = os.path.abspath('images')
os.makedirs(MOODBOARD_IMAGE_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=MOODBOARD_IMAGE_DIR), name="images")

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
                # Ensure image_url and json_url are properly formatted
                if item.get('image_url') and not item['image_url'].startswith('http'):
                    item['image_url'] = f"http://localhost:8000{item['image_url']}"
                if item.get('json_url') and not item['json_url'].startswith('http'):
                    item['json_url'] = f"http://localhost:8000{item['json_url']}"
                past_moodboard = Moodboard(**item)
                moodboards.append(past_moodboard)
    
    # Sort by date_created (most recent first)
    moodboards.sort(key=lambda x: x.date_created, reverse=True)
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
    # Clear all existing images before downloading new ones
    try:
        for file in os.listdir(MOODBOARD_IMAGE_DIR):
            if file.startswith("image_") and file.endswith(".png"):
                os.remove(os.path.join(MOODBOARD_IMAGE_DIR, file))
        print(f"Cleared existing images from {MOODBOARD_IMAGE_DIR}")
    except Exception as e:
        print(f"Warning: Could not clear existing images: {e}")
    
    # Download new images
    for i, image_url in enumerate(images.images):
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            with open(os.path.join(MOODBOARD_IMAGE_DIR, f"image_{i}.png"), "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        except requests.exceptions.RequestException as e:
            return StatusResponse(status_code=500, detail=e)
        except Exception as e:
            return StatusResponse(status_code=500, detail=e)
    return StatusResponse(status_code=200)

@app.post("/api/organize_moodboard")
def organize_moodboard(title: TitleResponse):
    ORGANIZER.title = title.title
    json_data = ORGANIZER.organize_images()
    return {"json_data": json_data}

@app.post("/api/save_moodboard")
async def save_moodboard(
    json_data: str = Form(...),
    image: UploadFile = File(...),
    title: str = Form(...),
    prompt: str = Form(...)
) -> SavedMoodboardResponse:
    try:
        print(f"Received save request: title={title}, prompt={prompt}")
        print(f"JSON data length: {len(json_data)}")
        print(f"JSON data preview: {json_data[:200]}...")
        
        # Generate unique ID for this moodboard
        moodboard_id = str(uuid.uuid4())
        
        # Sanitize title for filename (remove invalid characters)
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')  # Replace spaces with underscores
        
        # Save the PNG image
        image_filename = f"{safe_title}_{moodboard_id}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        print(f"Saving image to: {image_path}")
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Save the JSON placement data
        json_filename = f"{safe_title}_{moodboard_id}_placement.json"
        json_path = os.path.join('past_moodboards', 'jsons', json_filename)
        
        # Save the JSON placement data (no individual image copying)
        print(f"Saving JSON to: {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json.loads(json_data), f, ensure_ascii=False, indent=2)
        
        # Update the past_moodboards.json file
        past_moodboards_file = os.path.join('past_moodboards', 'past_moodboards.json')
        moodboard_entry = {
            "title": title,
            "prompt": prompt,
            "image_url": f"/past_moodboards/images/{image_filename}",
            "json_url": f"/past_moodboards/jsons/{json_filename}",
            "date_created": datetime.now().isoformat()  # Use precise timestamp
        }
        
        if os.path.exists(past_moodboards_file) and os.path.getsize(past_moodboards_file) > 0:
            with open(past_moodboards_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(moodboard_entry)
        
        with open(past_moodboards_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully saved moodboard: {json_filename}")
        return SavedMoodboardResponse(status="success", filename=json_filename)
        
    except Exception as e:
        print(f"Error saving moodboard: {str(e)}")
        import traceback
        traceback.print_exc()
        return SavedMoodboardResponse(status="error", filename=str(e))

