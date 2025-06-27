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

pipe_img = DiffusionPipeline.from_pretrained("")
pipe_img = pipe_img.to("cuda" if torch.cuda.is_available() else "cpu")


ntext =  """ในแม่น้ำสายหนึ่งมีจระเข้ชุกชุมถึงสามสายพันธุ์ด้วยกัน จึงทำให้ไม่มีใครกล้ามาจับปลา มีเพียงตาอยู่คนเดียวเท่านั้นที่คลุกคลีกับจระเข้และจับปลามาขายได้ เมื่อชาวบ้านเดือดร้อนที่ใช้แม่น้ำหล่อเลี้ยงชีวิตไม่ได้ เรื่องนี้จึงร้อนถึงหูพระราชา ตาอยู่จึงได้บอกกับพระราชาไปว่า ได้เลี้ยงจระเข้ตัวหนึ่งตั้งแต่ยังเล็กมันจึงไม่ทำร้าย ส่วนจระเข้ตัวอื่นถ้ามันกินอิ่มมันก็จะไม่ทำร้ายคน

พระราชาจึงได้มีพระราชโองการสั่งให้เสมียนไปนับจำนวนจระเข้เพื่อที่จะได้นำอาหารไปเลี้ยงพวกมันได้อย่างทั่วถึง เสมียนทั้งสามคนก็พยายามนับจระเข้ที่อยู่ทั้งบนบกและในน้ำ สุดท้ายก็นับจระเข้ได้คนละหนึ่งพันตัว รวมทั้งหมดมีจระเข้ถึงสามพันตัว และพระราชาก็ได้สั่งให้เลี้ยงอาหารจระเข้จนอิ่มและไม่ออกมาทำร้ายชาวบ้าน และหากินในแม่น้ำแห่งนี้ได้อย่างมีความสุข นิทานเรื่องนี้เป็นตำนานหรือนิทานพื้นบ้านของจังหวัดสุพรรณบุรี จนกลายมาเป็นชื่อตำบลจระเข้สามพันจนถึงทุกวันนี้"""

def processing_text(text):
    eng_text = GoogleTranslator(input='auto', output='en').translate(text)
    #elists = nltk.tokenize.sent_tokenize(eng_text, language='english')
    elists = nlp(eng_text).sents
    for i in elists:
        print(f'Create an illustration of : "{i.text.strip()}"')

    """
     
    """

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
    image.save("princess_sd35.png")
    
    print("done")
    #app = App(root)
    #root.mainloop()