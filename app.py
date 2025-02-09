
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
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()

# API Key configuration
OPENAI_API_KEY = "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"

def initialize_api_keys():
    """Initialize API key."""
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please set it in your environment variables.")
        return False
    openai.api_key = OPENAI_API_KEY
    return True

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
            max_pages_process = 10 # Limit to first 10 pages to avoid token limit issue
            num_pages = len(pdf)
            pages_to_process = min(num_pages, max_pages_process)

            for page_number in range(pages_to_process):
                page = pdf.get_page(page_number)
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
            max_pages_images = 5 # Limit to first 5 pages for images
            num_pages = len(pdf)
            pages_to_process = min(num_pages, max_pages_images)

            for page_number in range(pages_to_process):
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
            total_frames = int(duration * video.fps)

            # Limit frame extraction for very long videos to avoid processing too much text
            max_frames_process = 60 # Limit to 60 frames (approx 1 min if 1 fps)
            frame_interval = max(1, total_frames // max_frames_process) # Ensure interval is at least 1

            for i in range(0, total_frames, frame_interval):
                t = i / video.fps # Time in seconds
                frame = video.get_frame(t)
                # Convert frame to RGB format
                frame_rgb = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                frames.append(frame_rgb)

            return frames
        except Exception as e:
            st.error(f"Error extracting frames: {str(e)}")
            return []

    def process_video_content(self, video_file):
        """Process uploaded video content."""
        try:
            os.makedirs(self.temp_dir, exist_ok=True)

            # Save uploaded video file
            temp_video_path = os.path.join(self.temp_dir, "temp_video.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getvalue())

            if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                st.error("Invalid video file")
                return None

            # Load video using moviepy
            video = VideoFileClip(temp_video_path)

            # Get video duration and frames
            duration = video.duration
            frames = self._extract_frames(video)

            # Extract audio if present
            transcription_text = ""
            if video.audio is not None:
                audio_path = os.path.join(self.temp_dir, "temp_audio.mp3")
                video.audio.write_audiofile(audio_path)

                # Split and transcribe audio
                audio_chunk_paths = self._split_audio(audio_path)
                for chunk_path in audio_chunk_paths:
                    if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                        with open(chunk_path, "rb") as audio_file:
                            try:
                                transcription = openai.Audio.transcribe("whisper-1", audio_file)
                                transcription_text += transcription["text"]
                            except Exception as e:
                                st.warning(f"Audio transcription error: {str(e)}")

            # Save extracted frames as images
            image_paths = []
            frame_texts = []

            progress_bar = st.progress(0)
            total_frames = len(frames)

            for i, frame in enumerate(frames):
                # Save frame as image
                image_path = os.path.join(self.temp_dir, f"video_frame_{i}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)

                # Extract text from frame using OCR
                try:
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    text = pytesseract.image_to_string(frame_pil)
                    if text.strip():
                        frame_texts.append(text.strip())
                except Exception as e:
                    st.warning(f"OCR error on frame {i}: {str(e)}")

                # Update progress bar
                progress_bar.progress((i + 1) / total_frames)

            video.close()
            progress_bar.empty()

            # Create timing info for script generation
            timing_info = {
                "duration": duration,
                "frame_count": len(frames),
                "fps": video.fps
            }

            return {
                "transcription": transcription_text,
                "frame_texts": frame_texts,
                "image_paths": image_paths,
                "timing_info": timing_info
            }

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return None

    def enhance_script_generation(self, content, video_content=None):
        """Generate enhanced script to explain uploaded video content."""
        try:
            if video_content:
                # Create a detailed prompt for video narration
                video_prompt = f"""
                You are an AI narrator named Adoptly. Create a detailed script that explains this video demo.

                Video Information:
                - Duration: {video_content['timing_info']['duration']:.2f} seconds
                - Original Audio Transcription: {video_content['transcription'][:2000]} ... (summarized if too long)
                - Text Detected in Frames: {' '.join(video_content['frame_texts'][:100])} ... (summarized if too long)

                Create a natural, engaging script that:
                1. Introduces yourself as Adoptly at the start
                2. Describes what's happening in the video
                3. Explains any features or functionality shown
                4. Matches the video's timing and pacing
                5. Uses a friendly, professional tone

                Format the script with timestamps [00:00] at key points to match the video duration.
                Focus on creating a cohesive narrative that explains the demo clearly.
                """

                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert AI narrator for product demos."},
                        {"role": "user", "content": video_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )

                return response.choices[0].message.content

            # For non-video content
            intro_prompt = """
            Create a short, engaging introduction for a demo video, spoken by an AI persona named Adoptly. Adoptly should:
            1. Introduce itself by name
            2. State that it will be guiding the viewer through a demo
            3. Briefly mention the purpose of the demo
            4. Sound friendly, helpful, and enthusiastic
            """

            # Summarize content if it's too long to avoid token limits
            if len(content) > 5000: # Example limit, adjust as needed
                summary_prompt_for_script = f"""Summarize the following text to be concise and informative for generating a demo video script. Focus on the key topics and purpose. Limit the summary to around 500 words. Original Text: {content}"""
                summary_response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert summarizer for demo video scripts."},
                        {"role": "user", "content": summary_prompt_for_script}
                    ],
                    temperature=0.5,
                    max_tokens=700 # Adjust max_tokens for summary as needed
                )
                summary = summary_response.choices[0].message.content
            else:
                summary = content


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

            final_script_prompt = f"""
                You are an expert scriptwriter for product demo videos. Create a compelling script based on:
                Summary: {summary}
                """

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert scriptwriter for product demo videos."},
                    {"role": "user", "content": final_script_prompt}
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
            # Split by timestamp pattern [00:00] and periods
            parts = script.split('[')
            for part in parts:
                if ']' in part:
                    timestamp_text = part.split(']')
                    if len(timestamp_text) > 1:
                        text = timestamp_text[1]
                    else:
                        continue
                else:
                    text = part

                sentences = text.split('. ')
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
                clean_segment = segment.strip()
                if clean_segment:
                    tts = gTTS(text=clean_segment, lang='en', slow=False)
                    audio_path = os.path.join(self.temp_dir, f'audio_{i}.mp3')
                    tts.save(audio_path)
                    audio_files.append(audio_path)

            return audio_files
        except Exception as e:
            st.error(f"Error creating audio: {str(e)}")
            return None

    def create_video(self, image_paths, audio_files, background_path=None):
        """Create video from images and audio with enhanced error handling."""
        try:
            if not image_paths and not audio_files:
                st.error("No images or audio files available")
                return None

            num_clips = min(len(image_paths), len(audio_files))
            clips = []

            for i in range(num_clips):
                try:
                    if not os.path.exists(audio_files[i]):
                        st.warning(f"Audio file missing: {audio_files[i]}")
                        continue
                    if not os.path.exists(image_paths[i]):
                        st.warning(f"Image file missing: {image_paths[i]}")
                        continue

                    audio_clip = AudioFileClip(audio_files[i])
                    image_clip = ImageClip(image_paths[i]).set_duration(audio_clip.duration)

                    if background_path:
                        background_clip = ImageClip(background_path).set_duration(audio_clip.duration).resize(image_clip.size)
                        image_clip = CompositeVideoClip([background_clip, image_clip.set_pos("center")])

                    video_clip = image_clip.set_audio(audio_clip)
                    clips.append(video_clip)
                except Exception as e:
                    st.warning(f"Error creating clip {i}: {str(e)}")
                    continue

            if not clips:
                st.error("No valid clips could be created")
                return None

            final_clip = concatenate_videoclips(clips, method="compose")
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
                <p>{message}</p></div>
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
                <p>Supported formats: PDF, Video (MP4)</p>
            </div>
            """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Document (PDF) or Video (MP4)", type=['pdf', 'mp4'])
    

        if uploaded_file:
            try:
                doc_content = ""
                temp_file_path = None
                image_paths = []
                video_content = None
                audio_files = []
                script = None
                background_image_path = None


                # Process the uploaded file based on its type
                file_type = uploaded_file.type
                if file_type == 'application/pdf':
                    with st.spinner(""):
                        self.show_processing_animation("📄 Processing PDF document...")
                        temp_file_path = os.path.join(self.temp_dir, uploaded_file.name)
                        with open(temp_file_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())
                        doc_content = self.extract_text_from_pdf(temp_file_path)
                        image_paths = self.extract_images_from_pdf(temp_file_path)

                elif file_type == 'video/mp4':
                    with st.spinner(""):
                        self.show_processing_animation("🎥 Processing video content...")
                        video_content = self.process_video_content(uploaded_file)
                        if video_content is None:
                            st.error("Video processing failed")
                            return
                        image_paths = video_content.get('image_paths', [])

                # Generate script
                if doc_content or video_content:
                    with st.spinner(""):
                        self.show_processing_animation("🤖 Generating AI voiceover script...")
                        script = self.enhance_script_generation(doc_content, video_content)

                    if script:
                        with st.spinner(""):
                            self.show_processing_animation("🎤 Creating AI voiceover...")
                            audio_files = self.create_enhanced_audio(script)
                            if audio_files is None:
                                st.error("Audio creation failed")
                                return

                # Create final video
                if script and (image_paths or audio_files):
                    with st.spinner(""):
                        self.show_processing_animation("🎬 Creating final video...")
                        video_path = self.create_video(image_paths, audio_files, background_image_path)

                        if video_path and os.path.exists(video_path):
                            st.success("🎉 Your AI-narrated demo video is ready!")
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
