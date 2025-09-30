import os
from mistralai import Mistral
from backend.system_prompt import *
from backend.request_models import KeywordsResponse

class MBGen:
    def __init__(self, user_prompt, image=None):
        self.client = Mistral(api_key=os.getenv("MISTRAL_API"))
        self.llm_model = "mistral-medium-2505"

        self.user_prompt = user_prompt
        self.image = image

    def generate_title(self):
        chat_response = self.client.chat.complete(
            model = self.llm_model,
            messages = [
                {
                    "role": "system",
                    "content": get_title_prompt(),
                },
                {
                    "role": "user",
                    "content": self.user_prompt,
                }
            ],
            max_tokens=50
        )

        return chat_response.choices[0].message.content.strip().strip('"')
    
    def generate_keywords(self):
        user_content = [
            {
                "type": "text",
                "text": self.user_prompt,
            }
        ]

        if self.image:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{self.image}",
                }
            )

        chat_response = self.client.chat.complete(
            model = self.llm_model,
            messages = [
                {
                    "role": "system",
                    "content": get_generation_prompt(),
                },
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            response_format= {
                "type": "json_object",
                "schema": KeywordsResponse.model_json_schema(),
            },
            max_tokens=500
        )

        return chat_response.choices[0].message.content