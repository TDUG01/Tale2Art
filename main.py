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

pipe = StableDiffusionPipeline .from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype = torch.float16)
pipe = pipe.to("cuda")


if __name__ == "__main__":
    prompt = "A fantasy princess sitting in a golden castle, highly detailed, magical atmosphere"
    image = pipe(prompt=prompt).images[0]
    image.save("princess_sd35.png")
    
    print("done")