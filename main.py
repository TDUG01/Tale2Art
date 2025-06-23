import spacy
import openai
import gradio as gr
import os
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# setup
nlp = spacy.load("th_core_news_md")
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
