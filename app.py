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
from scipy.io.wavfile import write
import re

class EnhancedAdoptlyDemoCreator:
    def __init__(self):
        self.setup_streamlit()
        self.temp_dir = tempfile.mkdtemp()

    def setup_streamlit(self):
        """Initialize Streamlit UI configuration"""
        st.set_page_config(
            page_title="Adoptly Demo Creator",
            page_icon="🎥",
            layout="wide"
        )
        
        # Add your existing CSS styles here
        st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 10px;
            padding: 10px 25px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .upload-box {
            border: 2px dashed #FF4B4B;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .upload-box:hover {
            border-color: #ff7676;
            background-color: rgba(255,75,75,0.05);
        }
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 5px solid #28a745;
        }
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 5px solid #dc3545;
        }
        .status-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        .processing-animation {
            text-align: center;
            padding: 20px;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-radius: 50%;
            border-top: 5px solid #FF4B4B;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 10px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .header-container {
            background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    def handle_errors(func):
        """Decorator for error handling"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return None
        return wrapper

    @handle_errors
    def extract_text_from_pdf(self, pdf_file):
        """Enhanced PDF text extraction with improved formatting"""
        pdf = pdfium.PdfDocument(pdf_file)
        text_content = []
        
        for page in pdf:
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            
            # Clean and format the text
            text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
            text = text.replace('\n\n', '\n').strip()
            
            # Remove common unnecessary phrases
            text = re.sub(r'(Click here|See more|Read more|Learn more|Continue reading)\b', '', text)
            
            text_content.append(text)
        
        return "\n".join(text_content)

    @handle_errors
    def generate_script(self, content):
        """Generate natural, concise script without unnecessary words"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional video script writer creating concise, natural narrations. Focus on key information without filler words or unnecessary phrases."
                    },
                    {
                        "role": "user",
                        "content": f"""Create a natural, engaging video script that:
                        1. Focuses on core message and key points
                        2. Uses conversational language
                        3. Avoids unnecessary words and phrases
                        4. Maintains natural flow
                        5. Is optimized for voice narration
                        
                        Content to convert:
                        {content}"""
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            script = response['choices'][0]['message']['content']
            
            # Post-process script to remove any remaining filler words
            script = self.clean_script(script)
            
            return script
            
        except Exception as e:
            st.error(f"Error generating script: {str(e)}")
            return None

    def clean_script(self, script):
        """Clean script by removing unnecessary words and phrases"""
        # List of common filler words and phrases to remove
        filler_words = [
            r'\bbasically\b', r'\bactually\b', r'\bliterally\b',
            r'\bso,\s', r'\bwell,\s', r'\byou see,\s',
            r'\bas you can see\b', r'\bas mentioned\b',
            r'\bin conclusion\b', r'\bto sum up\b'
        ]
        
        # Remove filler words
        for filler in filler_words:
            script = re.sub(filler, '', script)
        
        # Clean up multiple spaces and newlines
        script = re.sub(r'\s+', ' ', script)
        script = re.sub(r'\n\s*\n', '\n\n', script)
        
        return script.strip()

    @handle_errors
    def create_audio(self, script):
        """Create audio with natural pacing and breaks"""
        if not script:
            return []
            
        audio_files = []
        
        # Split script into optimal chunks for natural narration
        chunks = self.split_into_natural_chunks(script)
        
        for i, chunk in enumerate(chunks):
            tts = gTTS(text=chunk, lang='en')
            audio_path = os.path.join(self.temp_dir, f'audio_{i}.mp3')
            tts.save(audio_path)
            
            # Add natural pause after each chunk
            self.add_pause_to_audio(audio_path)
            
            audio_files.append(audio_path)
            
        return audio_files

    def split_into_natural_chunks(self, script):
        """Split script into natural chunks for better narration"""
        # Split by sentences while preserving natural breaks
        sentences = re.split(r'(?<=[.!?])\s+', script)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > 300:  # Optimal chunk size
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def add_pause_to_audio(self, audio_path):
        """Add natural pause at the end of audio segments"""
        try:
            audio = AudioFileClip(audio_path)
            silence = AudioFileClip(self.create_silence(0.3))  # 0.3 second pause
            final_audio = concatenate_audioclips([audio, silence])
            final_audio.write_audiofile(audio_path)
        except Exception as e:
            st.warning(f"Error adding pause to audio: {str(e)}")

    def create_silence(self, duration):
        """Create silent pause"""
        silence_path = os.path.join(self.temp_dir, 'silence.mp3')
        sample_rate = 44100
        samples = np.zeros(int(duration * sample_rate))
        write(silence_path, sample_rate, samples.astype(np.float32))
        return silence_path

    @handle_errors
    def extract_images_from_pdf(self, pdf_file):
        """Extract and enhance images from PDF"""
        pdf = pdfium.PdfDocument(pdf_file)
        image_paths = []
        
        for i, page in enumerate(pdf):
            pil_image = page.render().to_pil()
            
            # Enhance image quality
            enhanced_image = self.enhance_image(pil_image)
            
            image_path = os.path.join(self.temp_dir, f'slide_{i}.png')
            enhanced_image.save(image_path)
            image_paths.append(image_path)
            
        return image_paths

    def enhance_image(self, image):
        """Enhance image quality for better presentation"""
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

    @handle_errors
    def create_video(self, image_paths, audio_files):
        """Create video with improved transitions and timing"""
        if not image_paths or not audio_files:
            return None
            
        clips = []
        total_duration = 0
        
        # Calculate total audio duration
        audio_durations = []
        for audio_file in audio_files:
            audio_clip = AudioFileClip(audio_file)
            audio_durations.append(audio_clip.duration)
            total_duration += audio_clip.duration
            audio_clip.close()
        
        # Calculate time per image
        time_per_image = total_duration / len(image_paths)
        
        for i, (img_path, duration) in enumerate(zip(image_paths, audio_durations)):
            image_clip = ImageClip(img_path)
            
            # Add transitions
            image_clip = image_clip.set_duration(duration)
            image_clip = image_clip.fadein(0.5).fadeout(0.5)
            
            clips.append(image_clip)
        
        # Combine video and audio
        final_video = concatenate_videoclips(clips)
        audio_clips = [AudioFileClip(af) for af in audio_files]
        final_audio = concatenate_audioclips(audio_clips)
        final_video = final_video.set_audio(final_audio)
        
        # Write final video
        output_path = os.path.join(self.temp_dir, 'final_video.mp4')
        final_video.write_videofile(output_path, fps=24, codec='libx264')
        
        return output_path

    def show_processing_animation(self, message):
        """Display processing animation with message"""
        st.markdown(f"""
        <div class="processing-animation">
            <div class="loader"></div>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)

    def main(self):
        """Main application flow"""
        # Header
        st.markdown("""
        <div class="header-container">
            <h1>🎥 Adoptly Demo Creator</h1>
            <p style="font-size: 1.2rem;">Transform your presentations into engaging video demos powered by AI</p>
        </div>
        """, unsafe_allow_html=True)

        # Features Section
        st.markdown("""
        <div style="display: flex; justify-content: space-around; margin: 2rem 0;">
            <div class="status-card" style="flex: 1; margin: 0 10px;">
                <h3>🤖 AI-Powered</h3>
                <p>Smart script generation using GPT-4</p>
            </div>
            <div class="status-card" style="flex: 1; margin: 0 10px;">
                <h3>🎯 Professional Voice</h3>
                <p>Natural text-to-speech conversion</p>
            </div>
            <div class="status-card" style="flex: 1; margin: 0 10px;">
                <h3>⚡ Fast Processing</h3>
                <p>Quick video generation</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # File Upload Section
        st.markdown("""
        <div class="upload-box">
            <h2>📤 Upload Your Presentation</h2>
            <p>Supported formats: PDF</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['pdf'])
        
        if uploaded_file:
            try:
                # Save uploaded file
                temp_file_path = os.path.join(self.temp_dir, uploaded_file.name)
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                # Content Extraction
                with st.spinner(""):
                    self.show_processing_animation("📄 Extracting content from your presentation...")
                    content = self.extract_text_from_pdf(temp_file_path)
                    if content:
                        st.markdown('<div class="success-message">✅ Content extracted successfully</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-message">❌ Failed to extract content</div>', unsafe_allow_html=True)
                        return

                # Script Generation
                with st.spinner(""):
                    self.show_processing_animation("🤖 Generating engaging script with AI...")
                    script = self.generate_script(content)
                    if script:
                        st.markdown('<div class="success-message">✅ Script generated successfully</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-message">❌ Failed to generate script</div>', unsafe_allow_html=True)
                        return

                # Audio Creation
                with st.spinner(""):
                    self.show_processing_animation("🎤 Creating professional voiceover...")
                    audio_files = self.create_audio(script)
                    if audio_files:
                        st.markdown('<div class="success-message">✅ Audio created successfully</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-message">❌ Failed to create audio</div>', unsafe_allow_html=True)
                        return

                # Slide Processing
                with st.spinner(""):
                    self.show_processing_animation("🖼️ Processing presentation slides...")
                    image_paths = self.extract_images_from_pdf(temp_file_path)
                    if image_paths:
                        st.markdown('<div class="success-message">✅ Slides processed successfully</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-message">❌ Failed to process slides</div>', unsafe_allow_html=True)
                        return

                # Video Creation
                with st.spinner(""):
                    self.show_processing_animation("🎬 Creating your video demo...")
                    video_path = self.create_video(image_paths, audio_files)

                if video_path and os.path.exists(video_path):
                    st.markdown("""
                    <div class="success-message" style="text-align: center;">
                        <h2>🎉 Your video demo is ready!</h2>
                        <p>Preview your video below and download when ready.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Video Preview
                    st.video(video_path)
                    
                    # Script Review and Download Options
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        with st.expander("📝 View Generated Script"):
                            st.markdown(f"""
                            <div class="status-card">
                                <h4>Generated Script:</h4>
                                <p style="white-space: pre-line;">{script}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        with open(video_path, 'rb') as f:
                            video_bytes = f.read()
                            st.download_button(
                                label="⬇️ Download Video Demo",
                                data=video_bytes,
                                file_name="adoptly_demo.mp4",
                                mime="video/mp4",
                            )
                        
                        st.markdown("""
                        <div class="status-card">
                            <h4>Video Details:</h4>
                            <ul>
                                <li>Format: MP4</li>
                                <li>Quality: HD</li>
                                <li>Audio: Stereo</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-message">❌ Failed to create video</div>', unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="error-message">❌ An error occurred: {str(e)}</div>', unsafe_allow_html=True)
                st.info("Please try again or contact support if the issue persists.")

        # Footer
        st.markdown("""
        <div style="text-align: center; margin-top: 50px; padding: 20px;">
            <p>Made with ❤️ by Adoptly | Need help? Contact support@adoptly.io</p>
        </div>
        """, unsafe_allow_html=True)

def initialize_api_keys():
    """Initialize API key using Streamlit secrets."""
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        return True
    except Exception as e:
        st.error("OpenAI API key not found in Streamlit secrets. Please add it to your secrets.toml file.")
        return False

if __name__ == "__main__":
    app = EnhancedAdoptlyDemoCreator()
    app.main()
