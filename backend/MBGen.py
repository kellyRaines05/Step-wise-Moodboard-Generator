import os
import moondream as md
from backend.system_prompt import *

class MBGen:
    def __init__(self, user_prompt, image=None):
        self.model = md.vl(api_key=os.getenv("MOONDREAM"))
        self.user_prompt = user_prompt
        self.image = image

    def generate_title(self):
        settings = {"max-tokens": 10}
        prompt = get_title_prompt() + "\nUser request: " + self.user_prompt
        if self.image is not None:
            result = self.model.query(image=self.image, question=prompt, settings=settings)
        else:
            result = self.model.query(question=prompt, settings=settings)
        return result["answer"].strip().replace("\"", "")
    
    def generate_keywords(self):
        settings = {"max-tokens": 100}
        prompt = get_generation_prompt() + "\nUser request: " + self.user_prompt
        if self.image is not None:
            result = self.model.query(image=self.image, question=prompt, settings=settings)
        else:
            result = self.model.query(question=prompt, settings=settings)
        return result["answer"].strip().replace("\"", "")