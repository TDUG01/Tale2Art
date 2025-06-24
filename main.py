import os
import gradio as gr
from pythainlp.tokenize import sent_tokenize
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from diffusers import DiffusionPipeline
import torch

# setup
load_dotenv()
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5", torch_dtype = torch.float16, variant = "fp16")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    prompt = "A fantasy princess sitting in a golden castle, highly detailed, magical atmosphere"
    image = pipe(prompt=prompt).images[0]
    image.save("princess_sd35.png")