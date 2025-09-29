from pydantic import BaseModel
from datetime import date as Date

class StatusResponse(BaseModel):
    status_code: int

class GenerationRequest(BaseModel):
    prompt: str

class Moodboard(BaseModel):
    title: str
    prompt: str
    date_created: Date
    image_url: str | None = None

class PastMoodboardsResponse(BaseModel):
    moodboards: list[Moodboard]

class TitleResponse(BaseModel):
    title: str

class KeyWordsResponse(BaseModel):
    keywords: list[str]

class SavedMoodboardResponse(BaseModel):
    status: str
    filename: str