system_prompt = '''

Moodboards inspire or communicate creative ideas for others to understand. For example, a wedding requires a lot of organization of different parts. The ceremony: location (inside or outside), decorations, dress code, and the procession and props needed such as a flower petals for the flower girl or how the exit will include different object like blowing bubbles, ringing bells, or throwing seeds. This is just an American wedding style. There could be many other traditions that the user may specify that they require different ideas.
A wedding also includes a reception, which is often where moodboards are created. Dinner includes tablecloths, napkins, plates, utensils, decorations, location (inside or outside), and if there will be dancing or other activities like a picture spot, there also needs to be a set up and inspiration how that will look.
This is just one example of where a moodboard is needed. It can also apply to fashion, coordinating different parts of an outfit(s) together. It can be used to organize a room's interior design with different furniture according to its purpose and theme. There are many uses for a moodboard in planning or creative outlet of expression.
'''

summarization_instructions = '''Create a title that captures the moodboard theme based on the user request. ONLY use up to 3 words.'''

keyword_instruction = '''What are some keywords that would be useful to search for images to include in a moodboard based on the user request? 

First answer the following questions to help you brainstorm relevant keywords:
1. What is the purpose of the moodboard (e.g., wedding, fashion, interior design, event planning, etc.)?
2. What descriptive words would fit the main theme of the moodboard such as colors, styles, patterns, etc.?
3. What are the appropriate images and items to be shown for the moodboard? For example,
    a. If it is fashion/apparel... What specific types of clothing fit this theme?
    b. If it is wedding... What specific types of flowers, decorations, or settings fit this theme?
    c. If it is interior design... What specific furniture or types of rooms fit this theme?

With these answers in mind, come up with 10-20 specific and descriptive keywords in a JSON comma-separated list that would be useful to search for images to include in the moodboard? These keywords should include colors, styles, and any other relevant details that describe the items. Only put 2-3 words per keyword.
If given an image, include keywords that describe the style, colors, and objects in the image.
'''

def get_generation_prompt():
    return keyword_instruction

def get_title_prompt():
    return summarization_instructions