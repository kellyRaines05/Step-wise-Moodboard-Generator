from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
from system_prompt import get_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path,
                                                torch_dtype=torch.bfloat16,
                                                _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager").to(DEVICE)
# example image input
image = Image.open("example_image.png")

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": get_prompt()}
        ],
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": " I am trying to make a moodboard for a fashion show. I want it to fit the desert theme: Fall/Winter time. What are some keywords I should look for that are a similar vibe to the reference image?"}
        ]
    },
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs

generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
