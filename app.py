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
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# API Key configuration
def initialize_api_keys():
    """Initialize API key using Streamlit secrets."""
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        return True
    except Exception as e:
        st.error("OpenAI API key not found in Streamlit secrets. Please add it to your secrets.toml file.")
        return False

class HumanizedAdoptlyDemoCreator:
    def __init__(self):
        self.setup_streamlit()
        self.temp_dir = tempfile.mkdtemp()

    def setup_streamlit(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Humanized Demo Creator",
            page_icon="🎥",
            layout="wide"
        )
        st.markdown("""
        <style>
        .processing-animation {
            text-align: center;
            padding: 2rem;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-radius: 50%;
            border-top: 4px solid #3498db;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)

    def setup_api_key(self):
        """Setup API key."""
        return initialize_api_keys()

    def extract_text_from_pdf(self, pdf_file):
        """Extract text content from PDF with improved formatting."""
        try:
            pdf = pdfium.PdfDocument(pdf_file)
            text_content = []
            max_pages_process = 10
            num_pages = len(pdf)
            pages_to_process = min(num_pages, max_pages_process)

            for page_number in range(pages_to_process):
                page = pdf.get_page(page_number)
                textpage = page.get_textpage()
                text = textpage.get_text_range()
                
                # Clean and format the text
                text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
                text = text.replace('\n\n', '\n').strip()
                text_content.append(text)

            return " ".join(text_content)
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None

    def extract_images_from_pdf(self, pdf_path):
        """Extract and enhance images from PDF."""
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            image_paths = []
            max_pages_images = 5
            num_pages = len(pdf)
            pages_to_process = min(num_pages, max_pages_images)

            for page_number in range(pages_to_process):
                page = pdf.get_page(page_number)
                pil_image = page.render().to_pil()
                
                # Enhance image quality
                enhanced_image = self.enhance_image(pil_image)
                
                image_path = os.path.join(self.temp_dir, f"image_{page_number}.png")
                enhanced_image.save(image_path, quality=95)
                image_paths.append(image_path)

            return image_paths
        except Exception as e:
            st.error(f"Error extracting images from PDF: {str(e)}")
            return []

    def enhance_image(self, image):
        """Enhance image quality for better video presentation."""
        try:
            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply enhancements
            enhanced = cv2.detailEnhance(cv_image, sigma_s=10, sigma_r=0.15)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            return Image.fromarray(enhanced)
        except Exception as e:
            st.warning(f"Image enhancement failed: {str(e)}")
            return image

    def humanize_script_generation(self, content, video_content=None):
        """Generate natural-sounding script without AI references."""
        try:
            # Extract key points from content
            key_points = self.extract_humanized_key_points(content)
            
            if video_content:
                duration = video_content['timing_info']['duration']
                segments = self.plan_video_segments(duration, key_points)
                
                video_prompt = f"""
                Create a natural, engaging narration for a product demo video that sounds completely human.
                
                Duration: {duration:.2f} seconds
                Key Points to Cover:
                {key_points}
                
                Segment Timing:
                {segments}
                
                Strict Guidelines:
                1. Absolutely NO mention of AI, technology, or any technical terms
                2. Use natural, conversational language like a human presenter
                3. Avoid any references to screens, interfaces, or digital elements
                4. Focus on benefits and real-world applications
                5. Use active voice and personal pronouns
                6. Include natural transitions between ideas
                7. Maintain consistent pacing with the timing plan
                
                Example Structure:
                "When we developed this solution, we focused on three key pillars: [benefits]. 
                Let me show you how this works in practice... [natural transition] 
                What really sets this apart is [unique value proposition]..."
                """
            else:
                words = len(content.split())
                estimated_duration = (words / 150) * 60
                segments = self.plan_video_segments(estimated_duration, key_points)
                
                video_prompt = f"""
                Create a natural narration for a {estimated_duration:.0f}-second product demo that sounds human.

                Key Points to Cover:
                {key_points}
                
                Segment Timing:
                {segments}
                
                Strict Requirements:
                1. NO technical jargon or AI references
                2. Sound like a friendly human expert
                3. Use storytelling techniques
                4. Focus on user benefits, not features
                5. Include rhetorical questions and natural pauses
                """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a skilled human presenter creating natural product demos."},
                    {"role": "user", "content": video_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            script = response.choices[0].message.content
            return self.post_process_script(script)

        except Exception as e:
            st.error(f"Error in script generation: {str(e)}")
            return None

    def extract_humanized_key_points(self, content):
        """Extract key points focusing on human benefits."""
        try:
            prompt = f"""
            Extract key points focusing on human benefits and real-world applications:
            
            1. Identify core human needs addressed
            2. List practical benefits (avoid technical specs)
            3. Highlight emotional value propositions
            4. Note any user success stories or testimonials
            
            Content:
            {content[:3000]}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You extract human-centric benefits from technical content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"Error extracting key points: {str(e)}")
            return content[:1000]

    def post_process_script(self, script):
        """Clean and format the generated script."""
        # Remove technical references
        script = re.sub(r'\b(AI|artificial intelligence|algorithm|system|interface|screen|user)\b', '', script, flags=re.IGNORECASE)
        
        # Remove multiple consecutive newlines
        script = re.sub(r'\n{3,}', '\n\n', script)
        
        # Ensure natural language patterns
        script = re.sub(r'\.\s([A-Z])', lambda m: '. ' + m.group(1).lower(), script)
        
        return script.strip()

    def create_natural_audio(self, script):
        """Create audio with human-like pacing."""
        try:
            segments = self.split_script_into_segments(script)
            
            audio_files = []
            for segment in segments:
                clean_text = self.clean_text_for_tts(segment)
                
                if clean_text:
                    tts = gTTS(text=clean_text, lang='en', slow=False)
                    audio_path = os.path.join(self.temp_dir, f'audio_{len(audio_files)}.mp3')
                    tts.save(audio_path)
                    
                    self.add_pause_to_audio(audio_path)
                    audio_files.append(audio_path)

            return audio_files
        except Exception as e:
            st.error(f"Error creating audio: {str(e)}")
            return None

    # Remaining methods stay similar with human-centric improvements

    def main(self):
        """Main application function."""
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%); border-radius: 15px; margin-bottom: 2rem;">
                <h1 style="color: white;">🎥 Humanized Demo Creator</h1>
                <p style="color: white; font-size: 1.2rem;">Create natural product demos that sound completely human</p>
            </div>
            """, unsafe_allow_html=True)

        if not self.setup_api_key():
            return

        st.markdown("""
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 2rem;">
                <h2>📤 Upload Your Content</h2>
                <p>Transform documents into natural-sounding demo videos</p>
                <p>Supported formats: PDF</p>
            </div>
            """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

        if uploaded_file:
            try:
                with st.expander("⚙️ Presentation Settings", expanded=False):
                    st.slider("Speaking Pace", 0.8, 1.2, 1.0, 0.1,
                             help="Adjust the speed of the narration")
                    st.checkbox("Add Natural Pauses", value=True,
                               help="Include human-like breathing pauses")
                    st.checkbox("Emotional Inflection", value=True,
                               help="Add subtle emotional variation")

                # Processing pipeline
                temp_file_path = os.path.join(self.temp_dir, uploaded_file.name)
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                with st.spinner("📄 Analyzing document..."):
                    doc_content = self.extract_text_from_pdf(temp_file_path)
                    image_paths = self.extract_images_from_pdf(temp_file_path)
                
                if doc_content:
                    with st.spinner("✍️ Crafting natural narration..."):
                        script = self.humanize_script_generation(doc_content)
                    
                    if script:
                        with st.spinner("🎙️ Generating voiceover..."):
                            audio_files = self.create_natural_audio(script)
                        
                        if audio_files:
                            with st.spinner("🎬 Producing final video..."):
                                video_path = self.create_video(image_paths, audio_files)
                            
                            if video_path:
                                st.success("✅ Demo Ready!")
                                st.video(video_path)
                                
                                with open(video_path, "rb") as f:
                                    st.download_button(
                                        "📥 Download Video",
                                        f,
                                        file_name="humanized_demo.mp4"
                                    )
                                
                                if st.button("📜 Show Transcript"):
                                    st.markdown(f"```\n{script}\n```")

            except Exception as e:
                st.error(f"Error creating demo: {str(e)}")

if __name__ == "__main__":
    app = HumanizedAdoptlyDemoCreator()
    app.main()
