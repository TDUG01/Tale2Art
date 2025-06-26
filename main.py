import os
import gradio as gr
import huggingface_hub
import torch
from pythainlp.tokenize import sent_tokenize
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline


# setup
load_dotenv()
api_key = os.getenv("API_KEY")
huggingface_hub.login(token=api_key)

text = """
ในแม่น้ำสายหนึ่งมีจระเข้ชุกชุมถึงสามสายพันธุ์ด้วยกัน จึงทำให้ไม่มีใครกล้ามาจับปลา มีเพียงตาอยู่คนเดียวเท่านั้นที่คลุกคลีกับจระเข้และจับปลามาขายได้ เมื่อชาวบ้านเดือดร้อนที่ใช้แม่น้ำหล่อเลี้ยงชีวิตไม่ได้ เรื่องนี้จึงร้อนถึงหูพระราชา ตาอยู่จึงได้บอกกับพระราชาไปว่า ได้เลี้ยงจระเข้ตัวหนึ่งตั้งแต่ยังเล็กมันจึงไม่ทำร้าย ส่วนจระเข้ตัวอื่นถ้ามันกินอิ่มมันก็จะไม่ทำร้ายคน

พระราชาจึงได้มีพระราชโองการสั่งให้เสมียนไปนับจำนวนจระเข้เพื่อที่จะได้นำอาหารไปเลี้ยงพวกมันได้อย่างทั่วถึง เสมียนทั้งสามคนก็พยายามนับจระเข้ที่อยู่ทั้งบนบกและในน้ำ สุดท้ายก็นับจระเข้ได้คนละหนึ่งพันตัว รวมทั้งหมดมีจระเข้ถึงสามพันตัว และพระราชาก็ได้สั่งให้เลี้ยงอาหารจระเข้จนอิ่มและไม่ออกมาทำร้ายชาวบ้าน และหากินในแม่น้ำแห่งนี้ได้อย่างมีความสุข นิทานเรื่องนี้เป็นตำนานหรือนิทานพื้นบ้านของจังหวัดสุพรรณบุรี จนกลายมาเป็นชื่อตำบลจระเข้สามพันจนถึงทุกวันนี้
"""

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    prompt = "A fantasy princess sitting in a golden castle, highly detailed, magical atmosphere"
    image = pipe(prompt=prompt, height=384, width=384).images[0]
    image.save("princess_sd35.png")
    
    print("done")