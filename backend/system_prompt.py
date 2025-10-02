system_prompt = '''You are a professional at creating moodboards for any topic, occasion, or purpose. 

Moodboards inspire or communicate creative ideas for others to understand. For example, a wedding requires a lot of organization of different parts. The ceremony: location (inside or outside), decorations, dress code, and the procession and props needed such as a flower petals for the flower girl or how the exit will include different object like blowing bubbles, ringing bells, or throwing seeds. This is just an American wedding style. There could be many other traditions that the user may specify that they require different ideas.
A wedding also includes a reception, which is often where moodboards are created. Dinner includes tablecloths, napkins, plates, utensils, decorations, location (inside or outside), and if there will be dancing or other activities like a picture spot, there also needs to be a set up and inspiration how that will look.
This is just one example of where a moodboard is needed. It can also apply to fashion, coordinating different parts of an outfit(s) together. It can be used to organize a room's interior design with different furniture according to its purpose and theme. There are many uses for a moodboard in planning or creative outlet of expression.
'''

summarization_instructions = '''Given the user request, give a title to the moodboard. 
It should summarize both the type/purpose (e.g. fashion, outfit, apparel, furniture, table, decorations, location, interior design, living room, event, wedding, etc.) and its theme. 
Do not make the title longer than 3 words.
'''

keyword_instruction = '''What are some relevant images and items that would be useful to include in a moodboard for the given user prompt?

First answer the following questions to help you brainstorm relevant keywords:
    1. What is the purpose of the moodboard (e.g. fashion, outfit, apparel, furniture, table, bedding, decorations, location, interior design, living room, event, wedding, etc.)?
    2. What descriptive words would fit the main theme of the moodboard? Could be things such as colors, styles, patterns, etc.
    3. What are the appropriate images and items to be shown for the moodboard? For example,
        a. If it is fashion/apparel, what garments should be selected? Skirt, pants, shirt, dress, etc.
        b. If it is wedding, what should the bride and groom wear? What should the place look like? Not only location, but what decor, what furniture, what dinning wear
        c. If it is interior design, what specific furniture does it need? Table, bed, lamp, couch, carpet, etc. What type of room is this?

With these answers in mind, come up with 5-10 items and/or descriptions of images to include in the moodboard. It should be a JSON-formatted list with the key "keywords".
These image descriptions should be able to answer all of the questions asked previously about the colors, styles, items or objects in the images in ONLY 2-3 words.

If given an image, include keywords that describe the style, colors, and objects in the image.
'''

combine_instruction = '''Given two lists of keywords, combine them into one list and remove any duplicates. Return the combined list in a JSON comma-separated format.'''

def get_generation_prompt():
    return keyword_instruction

def get_title_prompt():
    return summarization_instructions

def get_combine_prompt():
    return combine_instruction