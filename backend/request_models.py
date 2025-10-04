from pydantic import BaseModel

class StatusResponse(BaseModel):
    status_code: int
    detail: str | None = None

class GenerationRequest(BaseModel):
    prompt: str
    image_url: str | None = None

class ImageRequest(BaseModel):
    keyword: str

class DownloadRequest(BaseModel):
    images: list[str]

class Moodboard(BaseModel):
    title: str
    prompt: str
    image_url: str | None = None
    json_url: str | None = None
    date_created: str

class PastMoodboardsResponse(BaseModel):
    moodboards: list[Moodboard]

class TitleResponse(BaseModel):
    title: str

class KeywordsResponse(BaseModel):
    keywords: list[str]

class ImagesResponse(BaseModel):
    images: list[str]
    keyword: str

class SavedMoodboardResponse(BaseModel):
    status: str
    filename: str
