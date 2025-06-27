import os
import tkinter as tk
import huggingface_hub
import torch
import nltk
import spacy
import en_core_web_trf
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from diffusers import DiffusionPipeline


# setup
load_dotenv()
api_key = os.getenv("API_KEY")
huggingface_hub.login(token=api_key)
nlp = en_core_web_trf.load()

pipe_img = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe_img = pipe_img.to("cuda" if torch.cuda.is_available() else "cpu")


def processing_text(text):
    eng_text = GoogleTranslator(input='auto', output='en').translate(text)
    #elists = nltk.tokenize.sent_tokenize(eng_text, language='english')
    elists = nlp(eng_text).sents
    lists = [i.text.strip() for i in elists]
    return lists

def generate_image(text, number=1):
    prompt = f'Create an illustration of : "{text}"'
    image = pipe_img(prompt=prompt, height=512, width=512).images[0]
    image.save(f"page{number}.png")
        

class App:
    def __init__(self, root):
        self.root = root
        root.title("Tale2Art")
        root.geometry("800x600")
        
        self.label = tk.Label(root, text='Enter text:', width=50, height=5)
        self.label.pack()
        
        self.story = tk.Entry(root)
        self.story.pack()
        
        self.button = tk.Button(root, text='Generate', command=self.process_text)
        self.button.pack()
    def process_text(self):
        text = self.story.get()
        processing_text(text)


root = tk.Tk()
if __name__ == "__main__":
    prompt = 'Create an illustration of : "In one river, there were three species of crocodiles."'
    image = pipe_img(prompt=prompt, height=512, width=512).images[0]
    image.save("page1.png")
    
    print("done")
    #app = App(root)
    #root.mainloop()