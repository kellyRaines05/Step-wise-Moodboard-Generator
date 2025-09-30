from pydantic import BaseModel

class StatusResponse(BaseModel):
    status_code: int

class GenerationRequest(BaseModel):
    prompt: str
    image_url: str | None = None

class Moodboard(BaseModel):
    title: str
    prompt: str
    image_url: str | None = None
    date_created: str

class PastMoodboardsResponse(BaseModel):
    moodboards: list[Moodboard]

class TitleResponse(BaseModel):
    title: str

class KeywordsResponse(BaseModel):
    keywords: list[str]

class SavedMoodboardResponse(BaseModel):
    status: str
    filename: str
