import os
import tkinter as tk
import huggingface_hub
import torch
import en_core_web_trf
import time
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
    image.save(f"images/page{number}.png")
      
        
p_lists = []

class Input_App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tale2Art")
        
        self.label = tk.Label(self.root, text='Enter text:')
        self.label.pack()
        self.story = tk.Text(self.root, width=40, height=10)
        self.story.pack(ipadx=60, ipady=7)
        
        self.button = tk.Button(self.root, text='Generate', command=self.process_text)
        self.button.pack()
        
        self.root.mainloop()
    def process_text(self):
        text = self.story.get("1.0", tk.END)
        self.p_lists = processing_text(text)
        time.sleep(1)
        self.root.destroy()
        
class Process_App:
    def __init__(self):
        self.root = tk.Tk()
        self.p_lists = p_lists
        self.max = len(self.p_lists)
        self.now = 0
        self.root.title("Processing...")
        
        self.label = tk.Label(self.root, text=f"{self.p_lists[self.now]} | {self.now+1}/{self.max} Processing...")
        self.label.pack()
        self.process()
        
        if self.now == self.max:
            self.button = tk.Button(self.root, text='Finish', command=self.finish)
            self.button.pack()
        
        self.root.mainloop()
        
    def process(self):
        for i in range(self.max):
            generate_image(self.p_lists[i], self.now+1)
            self.now += 1
            self.label.config(text=f"{self.p_lists[self.now]} | {self.now+1}/{self.max} Processing...")
            self.root.update()
    
    def finish(self):
        self.root.destroy()
        

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    
    p_lists = Input_App().p_lists
    
    Process_App()
    
    print("Done")
    