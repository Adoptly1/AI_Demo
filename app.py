import streamlit as st
import os
from PIL import Image
import pypdfium2 as pdfium
import pandas as pd
from pathlib import Path
import time
from moviepy.editor import *
import openai
from gtts import gTTS
import tempfile
import base64
import pytesseract
import cv2
import numpy as np
from dotenv import load_dotenv
import io
import re

# API Key Configuration
def initialize_api_keys():
    """Initialize API key using Streamlit secrets."""
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        return True
    except Exception:
        st.error("OpenAI API key not found in Streamlit secrets. Please add it to your secrets.toml file.")
        return False

class AI_Demo_Creator:
    def __init__(self):
        self.setup_streamlit()
        self.temp_dir = tempfile.mkdtemp()

    def setup_streamlit(self):
        """Configure Streamlit page settings."""
        st.set_page_config(page_title="AI-Powered Demo Creator", page_icon="🎥", layout="wide")

    def setup_api_key(self):
        """Setup API key."""
        return initialize_api_keys()

    def extract_text_from_pdf(self, pdf_file):
        """Extract text content from PDF with improved formatting."""
        try:
            pdf = pdfium.PdfDocument(pdf_file)
            text_content = []
            for page_number in range(min(10, len(pdf))):  # Limit to 10 pages
                page = pdf.get_page(page_number)
                textpage = page.get_textpage()
                text = textpage.get_text_range().strip()
                text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
                text_content.append(text)
            return " ".join(text_content)
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None

    def extract_images_from_pdf(self, pdf_path):
        """Extract images from PDF."""
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            image_paths = []
            for page_number in range(min(5, len(pdf))):  # Limit to 5 images
                page = pdf.get_page(page_number)
                pil_image = page.render().to_pil()
                image_path = os.path.join(self.temp_dir, f"image_{page_number}.png")
                pil_image.save(image_path, quality=95)
                image_paths.append(image_path)
            return image_paths
        except Exception as e:
            st.error(f"Error extracting images from PDF: {str(e)}")
            return []

    def enhance_script_generation(self, content):
        """Generate AI script without unnecessary words or visuals."""
        try:
            prompt = f"""
            You are an expert product presenter. Create a smooth, engaging narration for an AI-generated product demo.
            
            **Instructions:**
            - Do **not** describe on-screen elements (e.g., "On the left, you see a person").
            - Focus on **benefits and storytelling**, not visual descriptions.
            - Use a **natural, human-like tone**.
            - Avoid robotic or overly formal language.
            - Structure content in a compelling, flowing manner.

            **Content to use:**
            {content[:3000]}  # Limit content to avoid token overflow
            
            **Output:**
            - A well-structured, human-like script.
            - No unnecessary timestamps, unless marking major sections.
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a professional presenter creating engaging demo scripts."},
                          {"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )

            return self.post_process_script(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error in script generation: {str(e)}")
            return None

    def post_process_script(self, script):
        """Refine AI script for a polished final version."""
        script = re.sub(r'\n{3,}', '\n\n', script)  # Remove excessive line breaks
        script = re.sub(r'\[\d{2}:\d{2}\]', '', script)  # Remove unnecessary timestamps
        script = script.replace('AI:', '').replace('Narrator:', '').strip()
        return script

    def create_audio(self, script):
        """Generate AI narration from the script."""
        try:
            tts = gTTS(text=script, lang='en', slow=False)
            audio_path = os.path.join(self.temp_dir, "narration.mp3")
            tts.save(audio_path)
            return audio_path
        except Exception as e:
            st.error(f"Error generating audio: {str(e)}")
            return None

    def create_video(self, image_paths, audio_path):
        """Generate a video with AI narration and smooth transitions."""
        try:
            if not image_paths or not audio_path:
                st.error("No images or audio available for video generation.")
                return None

            image_clips = [ImageClip(img).set_duration(3).fadein(0.5).fadeout(0.5) for img in image_paths]
            final_video = concatenate_videoclips(image_clips, method="compose")
            final_audio = AudioFileClip(audio_path)
            final_video = final_video.set_audio(final_audio)

            output_path = os.path.join(self.temp_dir, "final_video.mp4")
            final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
            return output_path
        except Exception as e:
            st.error(f"Error creating video: {str(e)}")
            return None

    def main(self):
        """Streamlit UI for AI-powered demo creation."""
        st.title("🎥 AI-Powered Demo Creator")

        if not self.setup_api_key():
            return

        uploaded_file = st.file_uploader("Upload PDF or MP4", type=['pdf', 'mp4'])
        if uploaded_file:
            with st.spinner("Processing file..."):
                temp_file_path = os.path.join(self.temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                if uploaded_file.type == 'application/pdf':
                    text_content = self.extract_text_from_pdf(temp_file_path)
                    image_paths = self.extract_images_from_pdf(temp_file_path)

                if text_content:
                    script = self.enhance_script_generation(text_content)
                    if script:
                        audio_path = self.create_audio(script)
                        if audio_path:
                            video_path = self.create_video(image_paths, audio_path)
                            if video_path and os.path.exists(video_path):
                                st.success("🎬 AI Demo Created Successfully!")
                                st.video(video_path)
                                st.download_button("⬇️ Download Video", open(video_path, "rb"), "AI_Demo.mp4", "video/mp4")

if __name__ == "__main__":
    app = AI_Demo_Creator()
    app.main()
