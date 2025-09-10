'''
This file takes the images in ./images and uses an algorithmic search function to find the layout with the best reward function.
Returns the JSON of the resultant placement of images with other relevant information

Rewards based on:
- image bounding box within bounds
- inverse image bounding box overlap with others
- pixel gap between borders of knn images
- image similarity grouping from CLIP embedding
- color grouping from avg in top left, right, bottom left, right so that similar corners are grouped together ?

Actions:
- swap images
- move image origin up/down/left/right 1 px

Issues:
- image scaling relative to the overall grid
- May need to make it stochastic to keep it from staying at local minima

Methodology:
- Arbitrarily add images to a grid
- Interatively run:
    - Calculate reward function
    - Randomly perform action
    - if it improves the reward function:
        - keep
    - else
        - revert, perform new action
'''

import os

#Need to import Image as IMG since tkinter has a similar Image open
from PIL import Image as IMG, ImageTk
from tkinter import *

#Define an image class to keep track of metadata
class image:
    #name
    filename = ""

    #relative x, y
    origin = (0,0)

    #length, width
    bounding_box = (0,0)
    
    #class def
    def __init__(self, filename, origin, bounding_box):
        self.origin = origin
        self.bounding_box = bounding_box
        self.filename = filename

#Define size of moodboard
length, width = 1000, 500
overall_img_scale = 0.5
root = Tk()
root.resizable(width=True, height=True)

#List to keep track of length of moodboard
img_list = []
img_refs = []

#Isolate title and palette
image_dir = os.listdir("./images")
img_title = image_dir.pop(image_dir.index("title.png"))
img_palette = image_dir.pop(image_dir.index("palette.png"))

#Arbitrary add to grid layout
curr_x = 0
curr_y = 0

max_img_per_row = 100

for i in range(len(image_dir)):
    img = image_dir[i]

    img_opened = IMG.open("./images/"+img)
    (width, height) = (int(img_opened.width*overall_img_scale), int(img_opened.height*overall_img_scale))
    img_opened = img_opened.resize((width, height))

    img_tk = ImageTk.PhotoImage(img_opened)
    image_label = Label(root, image=img_tk)

    #TODO: Replace this with placing pixels rather than grid layout since grid layout pixels aren't tracked
    image_label.grid(row=curr_y, column=curr_x)

    img_list.append(image(img, (0,0), img_opened.size))

    #Vital to prevent automatic garbage collection
    img_refs.append(img_tk)

    curr_x+=1
    if curr_x>max_img_per_row:
        curr_y+=1
        curr_x=0

root.mainloop()
