from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
from backend.system_prompt import *

class MBGen:
    def __init__(self, user_prompt, image=None):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_path = "HuggingFaceTB/SmolVLM-Base"
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_path,
                                                                 torch_dtype=torch.bfloat16,
                                                                 _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager").to(DEVICE)
        self.user_prompt = user_prompt
        self.image = image

    def generate_title(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": get_title_prompt()}
                ],
                "role": "user",
                "content": [
                    {"type": "image"} if self.image is not None else {},
                    {"type": "text", "text": self.user_prompt}
                ]
            },
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        if self.image is not None:
            inputs = self.processor(text=prompt, images=[self.image], return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, return_tensors="pt")
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts[0]
    
    def generate_keywords(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": get_generation_prompt()}
                ],
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.user_prompt}
                ]
            },
        ]

        if self.image is not None:
            inputs = self.processor(text=prompt, images=[self.image], return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, return_tensors="pt")
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts[0]
