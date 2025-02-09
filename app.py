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

# API Key Configuration
def initialize_api_keys():
    """Initialize API key using Streamlit secrets."""
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        return True
    except Exception:
        st.error("OpenAI API key not found in Streamlit secrets. Please add it to your secrets.toml file.")
        return False

class EnhancedAdoptlyDemoCreator:
    def __init__(self):
        self.setup_streamlit()
        self.temp_dir = tempfile.mkdtemp()

    def setup_streamlit(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Enhanced Adoptly Demo Creator",
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
        """Extract text content from PDF."""
        try:
            pdf = pdfium.PdfDocument(pdf_file)
            text_content = []

            for page in pdf:
                textpage = page.get_textpage()
                text_content.append(textpage.get_text_range())

            return "\n".join(text_content)
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None

    def extract_images_from_pdf(self, pdf_path):
        """Extract images from a PDF file."""
        image_paths = []
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            for page_number in range(len(pdf)):
                page = pdf.get_page(page_number)
                pil_image = page.render().to_pil()
                image_path = os.path.join(self.temp_dir, f"image_{page_number}.png")
                pil_image.save(image_path)
                image_paths.append(image_path)
            return image_paths
        except Exception as e:
            st.error(f"Error extracting images from PDF: {str(e)}")
            return []

    def _split_audio(self, audio_path, max_size_mb=24):
        """Split audio into smaller chunks."""
        try:
            chunk_paths = []
            audio = AudioFileClip(audio_path)
            total_duration = audio.duration
            max_size_bytes = max_size_mb * 1024 * 1024

            start_time = 0
            chunk_index = 0
            while start_time < total_duration:
                chunk_path = os.path.join(self.temp_dir, f"temp_audio_chunk_{chunk_index}.mp3")
                estimated_chunk_duration = (max_size_bytes / os.path.getsize(audio_path)) * total_duration * 0.9
                end_time = min(start_time + estimated_chunk_duration, total_duration)

                subclip = audio.subclip(start_time, end_time)
                subclip.write_audiofile(chunk_path, codec='libmp3lame')

                chunk_paths.append(chunk_path)
                start_time = end_time
                chunk_index += 1
            
            audio.close()
            return chunk_paths
        except Exception as e:
            st.error(f"Error splitting audio: {str(e)}")
            return []

    def _extract_frames(self, video):
        """Extract frames from video at regular intervals."""
        frames = []
        try:
            duration = video.duration
            fps = video.fps
            frame_interval = max(1, int(fps))  # Take at least 1 frame per second
            
            for t in np.arange(0, duration, 1.0/frame_interval):
                frame = video.get_frame(t)
                frames.append(frame)
                
            return frames
        except Exception as e:
            st.error(f"Error extracting frames: {str(e)}")
            return []

    def _process_frames(self, frames):
        """Process extracted frames for OCR with progress bar."""
        frame_texts = []
        if not frames:
            return frame_texts

        progress_bar = st.progress(0)
        num_frames = len(frames)

        for i, frame in enumerate(frames):
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_pil = Image.fromarray(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
                
                text = pytesseract.image_to_string(frame_pil)
                if text.strip():
                    frame_texts.append(text.strip())
                    
                progress_bar.progress((i + 1) / num_frames)
                
            except Exception as e:
                st.warning(f"Error processing frame {i}: {str(e)}")
                continue

        progress_bar.empty()
        return frame_texts

    def process_video_content(self, video_file):
        """Process uploaded video content."""
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            
            temp_video_path = os.path.join(self.temp_dir, "temp_video.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getvalue())

            if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                st.error("Invalid video file")
                return None

            video = VideoFileClip(temp_video_path)
            
            # Handle video without audio
            if video.audio is None:
                st.warning("No audio found in video")
                frames = self._extract_frames(video)
                return {
                    "transcription": "",
                    "frame_texts": self._process_frames(frames)
                }

            # Process audio
            audio_path = os.path.join(self.temp_dir, "temp_audio.mp3")
            video.audio.write_audiofile(audio_path)

            if not os.path.exists(audio_path):
                st.error("Failed to extract audio")
                return None

            # Process audio chunks
            audio_chunk_paths = self._split_audio(audio_path)
            if not audio_chunk_paths:
                st.error("Failed to split audio")
                return None

            # Transcribe audio
            transcription_text = ""
            for chunk_path in audio_chunk_paths:
                if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                    with open(chunk_path, "rb") as audio_file:
                        try:
                            transcription = openai.Audio.transcribe("whisper-1", audio_file)
                            transcription_text += transcription["text"]
                        except Exception as e:
                            st.error(f"Transcription error: {str(e)}")
                            return None

            # Extract and process frames
            frames = self._extract_frames(video)
            frame_texts = self._process_frames(frames)

            video.close()

            return {
                "transcription": transcription_text,
                "frame_texts": frame_texts
            }

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return None

    def enhance_script_generation(self, content, video_content=None):
        """Generate enhanced script with Adoptly introduction."""
        try:
            full_content = content
            if video_content:
                full_content += f"\n\nVideo Transcription:\n{video_content['transcription']}"
                full_content += f"\n\nScreen Text:\n{' '.join(video_content['frame_texts'])}"

            intro_prompt = """
            Create a short, engaging introduction for a demo video, spoken by an AI persona named Adoptly. Adoptly should:
            1. Introduce itself by name
            2. State that it will be guiding the viewer through a demo of the Adoptly platform
            3. Briefly mention the purpose of the demo
            4. Sound friendly, helpful, and enthusiastic
            5. Set a positive and exciting tone
            """

            initial_summary = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Create a concise summary..."},
                    {"role": "user", "content": full_content}
                ],
                temperature=0.7,
                max_tokens=300
            )
            summary = initial_summary.choices[0].message.content

            intro_response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": intro_prompt},
                    {"role": "user", "content": summary}
                ],
                temperature=0.7,
                max_tokens=150
            )
            adoptly_intro = intro_response.choices[0].message.content

            main_script_prompt = f"""
            You are an expert scriptwriter for product demo videos. Create a compelling script based on:
            Summary: {summary}
            Detailed Content: {full_content}
            """

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert scriptwriter..."},
                    {"role": "user", "content": main_script_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            main_script = response.choices[0].message.content
            return f"{adoptly_intro}\n\n{main_script}"

        except Exception as e:
            st.error(f"Error in script generation: {str(e)}")
            return None

    def create_enhanced_audio(self, script):
        """Create enhanced audio using gTTS."""
        try:
            segments = []
            current_segment = ""
            sentences = script.split('. ')

            for sentence in sentences:
                if len(current_segment + sentence) + 1 < 500:
                    current_segment += sentence + '. '
                else:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = sentence + '. '

            if current_segment:
                segments.append(current_segment)

            audio_files = []
            for i, segment in enumerate(segments):
                clean_segment = segment.replace('[pause]', '').replace('*', '')
                tts = gTTS(text=clean_segment, lang='en', slow=False)
                audio_path = os.path.join(self.temp_dir, f'audio_{i}.mp3')
                tts.save(audio_path)
                audio_files.append(audio_path)

            return audio_files
        except Exception as e:
            st.error(f"Error creating audio: {str(e)}")
            return None

    def _create_placeholder_images(self, count):
        """Create placeholder images if no images are available."""
        placeholder_paths = []
        for i in range(count):
            path = os.path.join(self.temp_dir, f'placeholder_{i}.png')
            img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
            cv2.putText(img, f'Slide {i+1}', (480, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            cv2.imwrite(path, img)
            placeholder_paths.append(path)
        return placeholder_paths

    def _create_silent_video(self, image_paths):
        """Create silent video from images."""
        clips = [ImageClip(img_path).set_duration(3) for img_path in image_paths]
        final_clip = concatenate_videoclips(clips)
        output_path = os.path.join(self.temp_dir, 'final_video.mp4')
        final_clip.write_videofile(output_path, fps=24, codec='libx264', 
                                 preset="medium", bitrate="5000k")
        return output_path

    def create_video(self, image_paths, audio_files):
        """Create video from images and audio with enhanced error handling."""
        try:
            if not image_paths and not audio_files:
                st.error("No images or audio files available")
                return None

            # If no images but have audio, create placeholder images
            if not image_paths and audio_files:
                image_paths = self._create_placeholder_images(len(audio_files))

            # If no audio but have images, create silent video
            if not audio_files and image_paths:
                return self._create_silent_video(image_paths)

            num_clips = min(len(image_paths), len(audio_files))
            clips = []

            for i in range(num_clips):
                try:
                    if not os.path.exists(audio_files[i]) or not os.path.exists(image_paths[i]):
                        continue

                    audio_clip = AudioFileClip(audio_files[i])
                    image_clip = ImageClip(image_paths[i]).set_duration(audio_clip.duration)
                    video_clip = image_clip.set_audio(audio_clip)
                    clips.append(video_clip)
                except Exception as e:
                    st.warning(f"Error creating clip {i}: {str(e)}")
                    continue

            if not clips:
                st.error("No valid clips could be created")
                return None

            final_clip = concatenate_videoclips(clips)
            output_path = os.path.join(self.temp_dir, 'final_video.mp4')
            final_clip.write_videofile(output_path, fps=24, codec='libx264', 
                                     preset="medium", bitrate="5000k")

            return output_path

        except Exception as e:
            st.error(f"Error creating video: {str(e)}")
            return None

    def show_processing_animation(self, message):
        """Display processing animation."""
        st.markdown(f"""
            <div class="processing-animation">
                <div class="loader"></div>
                <p>{message}</p>
            </div>
            """, unsafe_allow_html=True)

    def main(self):
        """Main application function."""
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%); border-radius: 15px; margin-bottom: 2rem;">
                <h1 style="color: white;">🎥 Adoptly Demo Creator</h1>
                <p style="color: white; font-size: 1.2rem;">Transform your content into engaging video demos powered by AI</p>
            </div>
            """, unsafe_allow_html=True)

        if not self.setup_api_key():
            return

        st.markdown("""
            <div class="upload-box">
                <h2>📤 Upload Your Content</h2>
                <p>Supported formats: PDF, PowerPoint (PPT, PPTX), Video (MP4)</p>
            </div>
            """, unsafe_allow_html=True)

        doc_file = st.file_uploader("Upload Document", type=['pdf', 'ppt', 'pptx'])
        video_file = st.file_uploader("Upload Demo Video (optional)", type=['mp4'])

        if doc_file or video_file:
            try:
                doc_content = ""
                temp_doc_path = None
                image_paths = []
                video_content = None
                audio_files = []
                script = None

                # Process document if uploaded
                if doc_file:
                    with st.spinner(""):
                        self.show_processing_animation("📄 Processing document...")
                        temp_doc_path = os.path.join(self.temp_dir, doc_file.name)
                        with open(temp_doc_path, 'wb') as f:
                            f.write(doc_file.getvalue())
                        doc_content = self.extract_text_from_pdf(temp_doc_path)
                        
                        with st.spinner("Extracting Images"):
                            image_paths = self.extract_images_from_pdf(temp_doc_path)
                            if not image_paths:
                                st.warning("No images extracted from document. Will create placeholders if needed.")

                # Process video if uploaded
                if video_file:
                    with st.spinner(""):
                        self.show_processing_animation("🎥 Processing video...")
                        video_content = self.process_video_content(video_file)
                        if video_content is None:
                            st.error("Video processing failed")
                        
                # Generate script
                if doc_content or video_content:
                    with st.spinner(""):
                        self.show_processing_animation("🤖 Generating enhanced script...")
                        script = self.enhance_script_generation(doc_content or "", video_content)

                    if script:
                        with st.spinner("Creating audio..."):
                            audio_files = self.create_enhanced_audio(script)
                            if audio_files is None:
                                st.error("Audio creation failed")
                                return

                # Create final video
                if script and (image_paths or audio_files):
                    with st.spinner("Creating Video..."):
                        video_path = self.create_video(image_paths, audio_files)

                        if video_path and os.path.exists(video_path):
                            st.success("🎉 Your video demo is ready!")
                            st.video(video_path)

                            with open(video_path, "rb") as file:
                                st.download_button(
                                    label="⬇️ Download Video",
                                    data=file,
                                    file_name="adoptly_demo.mp4",
                                    mime="video/mp4"
                                )

                            with st.expander("📝 View Generated Script"):
                                st.markdown(script)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

if __name__ == "__main__":
    app = EnhancedAdoptlyDemoCreator()
    app.main()


