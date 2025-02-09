import streamlit as st
import os
from PIL import Image
import pypdfium2 as pdfium
# pandas and pathlib aren't directly used, but are good practice for file handling
# so I'm leaving them.  Remove if you're *sure* you don't need them.
import pandas as pd
from pathlib import Path
import time
from moviepy.editor import *
import openai
from gtts import gTTS
import tempfile
import base64  # Not used directly, but can be useful for image embedding.  Keep.
import pytesseract
import cv2
import numpy as np
from dotenv import load_dotenv
import io
import re
from scipy.io.wavfile import write

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

    def process_video_content(self, video_file):
        """Processes the uploaded video file to extract frames and duration."""
        try:
            # Save the video file temporarily
            temp_video_path = os.path.join(self.temp_dir, video_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getbuffer())  # Use getbuffer() for BytesIO

            # Extract frames and duration using moviepy
            video_clip = VideoFileClip(temp_video_path)
            duration = video_clip.duration
            fps = video_clip.fps
            num_frames = int(duration * fps)

            image_paths = []
            # Extract a limited number of frames (e.g., 5)
            max_frames = 5
            for i in range(min(max_frames, num_frames)):
                frame_time = i * (duration / min(max_frames, num_frames))
                frame = video_clip.to_ImageClip(frame_time).to_array()
                frame_image = Image.fromarray(frame)
                enhanced_frame = self.enhance_image(frame_image)
                image_path = os.path.join(self.temp_dir, f"frame_{i}.png")
                enhanced_frame.save(image_path)
                image_paths.append(image_path)

            video_clip.close()  # Close the video clip after use

            return {
                'image_paths': image_paths,
                'timing_info': {'duration': duration}
            }

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return None



    def enhance_script_generation(self, content, video_content=None):
        """Generate script with a self-introduction and then the product demo."""
        try:
            # Consistent self-introduction.
            intro = "Hello! I'm Adoptly Demo Creator, your AI-powered assistant for creating engaging product demos.  I've analyzed the content you provided, and I'm ready to present a compelling demonstration.  Let's get started!"

            # Extract key points.
            key_points = self.extract_key_points(content)

            if video_content:
                duration = video_content['timing_info']['duration']
                segments = self.plan_video_segments(duration, key_points)
                video_prompt = f"""
                {intro}

                Now, let's craft a narration for the demo video:

                Duration: {duration:.2f} seconds
                Key Points:
                {key_points}

                Segment Timing:
                {segments}

                Guidelines:
                1. Start with the core value proposition.
                2. Use natural, conversational language.
                3. Smooth transitions between topics.
                4. Match pacing to segment timings.
                5. Focus on benefits and impact.

                Create a flowing narrative. Use minimal timestamps, only marking major transitions with [MM:SS].
                """
            else:  # PDF content
                words = len(content.split())
                estimated_duration = (words / 150) * 60
                segments = self.plan_video_segments(estimated_duration, key_points)
                video_prompt = f"""
                {intro}

                Now, here's a narration for a {estimated_duration:.0f}-second product demo based on the document:

                Key Points:
                {key_points}

                Segment Timing:
                {segments}

                Guidelines:
                1. Focus on core benefits and value.
                2. Use natural, flowing language.
                3. Clear transitions between topics.
                4. Match pacing to segment timings.
                5. Sound conversational and engaging.

                Create a natural narration. Use minimal timestamps, only marking major transitions with [MM:SS].
                """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert product presenter."},
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


    def extract_key_points(self, content):
        """Extract key points from content for better script structure."""
        try:
            prompt = f"""
            Extract the key points from this content, focusing on:
            1. Core value proposition
            2. Main benefits
            3. Key features
            4. Important metrics or statistics
            
            Content:
            {content[:3000]}  # Limit content length for token constraints
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting key points from product documentation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"Error extracting key points: {str(e)}")
            return content[:1000]  # Fallback to truncated content

    def plan_video_segments(self, duration, key_points):
        """Plan video segments with optimal timing."""
        try:
            prompt = f"""
            Create a timing plan for a {duration:.0f}-second video covering these key points:
            {key_points}
            
            Break the content into logical segments with timestamps, ensuring:
            1. Proper pacing for each topic
            2. Natural transitions
            3. Time for viewer comprehension
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at planning video timing and pacing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"Error planning segments: {str(e)}")
            return f"Divide content evenly across {duration:.0f} seconds"


    def post_process_script(self, script):
        """Clean and format the generated script."""
        # Remove multiple consecutive newlines
        script = re.sub(r'\n{3,}', '\n\n', script)
        
        # Ensure timestamps are properly formatted
        script = re.sub(r'\[(\d+):(\d+)\]', lambda m: f'[{int(m.group(1)):02d}:{int(m.group(2)):02d}]', script)
        
        # Clean up any remaining artifacts
        script = script.replace('AI:', '').replace('Narrator:', '').strip()
        
        return script

    def create_enhanced_audio(self, script):
        """Create enhanced audio, fixing repetition by splitting on sentences."""
        try:
            # Split script into natural segments based on *sentences*, not just timestamps.
            segments = self.split_script_into_segments(script)

            audio_files = []
            for segment in segments:
                # Clean the segment text
                clean_text = self.clean_text_for_tts(segment)

                if clean_text:
                    # Create audio
                    tts = gTTS(text=clean_text, lang='en', slow=False)
                    audio_path = os.path.join(self.temp_dir, f'audio_{len(audio_files)}.mp3')
                    tts.save(audio_path)

                    # Add slight pause after each segment
                    self.add_pause_to_audio(audio_path)

                    audio_files.append(audio_path)

            return audio_files
        except Exception as e:
            st.error(f"Error creating audio: {str(e)}")
            return None


    def split_script_into_segments(self, script):
        """Split script into natural segments based on *sentences* for TTS."""
        # Split by sentences.  This is the KEY to avoiding repetition.
        sentences = re.split(r'(?<=[.!?])\s+', script.strip())
        return [s for s in sentences if s.strip()]


    def clean_text_for_tts(self, text):
        """Clean and format text for optimal TTS output."""
        # Remove special characters and formatting
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add breaks at punctuation
        text = text.replace('.', '. ').replace('!', '! ').replace('?', '? ')
        
        return text

    def add_pause_to_audio(self, audio_path):
        """Add a natural pause at the end of audio segments."""
        try:
            audio = AudioFileClip(audio_path)
            pause = AudioFileClip(self.create_pause(0.3))  # 0.3 second pause
            final_audio = concatenate_audioclips([audio, pause])
            final_audio.write_audiofile(audio_path)
        except Exception as e:
            st.warning(f"Error adding pause to audio: {str(e)}")

    def create_pause(self, duration):
        """Create a silent pause of specified duration."""
        pause_path = os.path.join(self.temp_dir, 'pause.mp3')
        silence = np.zeros(int(duration * 44100))  # 44100 Hz sample rate
        write(pause_path, 44100, silence.astype(np.float32))
        return pause_path
    def create_video(self, image_paths, audio_files, background_path=None):
        """Create video, aligning image durations with audio segments."""
        try:
            if not image_paths or not audio_files:
                st.error("No images or audio files available")
                return None

            clips = []
            total_duration = 0

            # Calculate total audio duration
            for audio_file in audio_files:
                audio_clip = AudioFileClip(audio_file)
                total_duration += audio_clip.duration
                audio_clip.close()  # Close each clip after getting duration

            # Calculate time per image, *but don't use it directly for each clip's duration*
            time_per_image = total_duration / len(image_paths) if len(image_paths) > 0 else 0  # Avoid division by zero

            audio_clip_index = 0
            current_audio_time = 0

            for i, image_path in enumerate(image_paths):
                image_clip = ImageClip(image_path)

                # Determine the duration for *this* image clip based on the audio.
                if audio_clip_index < len(audio_files):
                    audio_clip = AudioFileClip(audio_files[audio_clip_index])
                    image_duration = audio_clip.duration
                    audio_clip.close() # Close after use
                else:  # Fallback, should not normally happen if audio matches images
                    image_duration = time_per_image

                image_clip = image_clip.set_duration(image_duration)

                # Resize images to maintain aspect ratio and fit within a standard frame (e.g., 1280x720)
                target_width = 1280
                target_height = 720
                image_clip = image_clip.resize(lambda t: (target_width, target_height) if image_clip.w > target_width or image_clip.h > target_height else (image_clip.w, image_clip.h) ) # Resize to fit


                 # Add fade in/out effects
                image_clip = image_clip.fadein(0.5).fadeout(0.5)

                # Center the image.  Important after resizing.
                image_clip = image_clip.set_pos("center")
                clips.append(image_clip)

                current_audio_time += image_duration
                if current_audio_time >= sum(AudioFileClip(af).duration for af in audio_files[:audio_clip_index+1]):
                    audio_clip_index += 1


            # Combine video clips
            final_video = concatenate_videoclips(clips, method="compose")

            # Combine audio files
            audio_clips = [AudioFileClip(af) for af in audio_files]
            final_audio = concatenate_audioclips(audio_clips)


            # Set audio to video, ensuring audio and video durations are aligned
            final_video = final_video.set_audio(final_audio)
            if final_video.duration > final_audio.duration:
                final_video = final_video.subclip(0, final_audio.duration) # Trim video if needed
            elif final_audio.duration > final_video.duration:
                #This part of the code is for creating a silence and adding to end
                #to adjust audio length
                difference_duration = final_audio.duration - final_video.duration
                pause_audio_path = self.create_pause(difference_duration)
                pause_audio_clip = AudioFileClip(pause_audio_path)
                list_for_concatinate = [final_audio,pause_audio_clip]
                final_audio = concatenate_audioclips(list_for_concatinate)
                final_video = final_video.set_audio(final_audio)
                #final_audio = final_audio.subclip(0, final_video.duration) # Trim audio *shouldn't* happen, but is a safeguard


            # Write final video
            output_path = os.path.join(self.temp_dir, 'final_video.mp4')
            final_video.write_videofile(output_path, fps=24, codec='libx264',
                                      preset="medium", bitrate="5000k",
                                      audio_codec='aac')

            # Clean up
            final_video.close()
            final_audio.close()

            for clip in audio_clips:
                clip.close()
            for clip in clips:
                clip.close()


            return output_path

        except Exception as e:
            st.error(f"Error creating video: {str(e)}")
            return None



    def show_processing_animation(self, message):
        """Display processing animation with progress updates."""
        st.markdown(f"""
            <div class="processing-animation">
                <div class="loader"></div>
                <p>{message}</p>
            </div>
            """, unsafe_allow_html=True)

    def main(self):
        """Main application function with improved UI and processing flow."""
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%); border-radius: 15px; margin-bottom: 2rem;">
                <h1 style="color: white;">🎥 Adoptly Demo Creator</h1>
                <p style="color: white; font-size: 1.2rem;">Create engaging product demos with natural AI narration</p>
            </div>
            """, unsafe_allow_html=True)

        if not self.setup_api_key():
            return

        st.markdown("""
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 2rem;">
                <h2>📤 Upload Your Content</h2>
                <p>Transform your PDF or video into an engaging demo with natural narration</p>
                <p>Supported formats: PDF, MP4</p>
            </div>
            """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'mp4'])

        if uploaded_file:
            try:
                with st.expander("⚙️ Processing Settings", expanded=False):
                    voice_speed = st.slider("Narration Speed", 0.8, 1.2, 1.0, 0.1)  # Not currently used, but could be!
                    enable_transitions = st.checkbox("Enable Visual Transitions", value=True)  # Already handled in create_video
                    high_quality = st.checkbox("High Quality Processing", value=True)  # Could control bitrate, etc.

                doc_content = ""
                temp_file_path = None
                image_paths = []
                video_content = None
                audio_files = []
                script = None

                # Process uploaded file
                file_type = uploaded_file.type
                progress_text = st.empty()
                progress_bar = st.progress(0)

                if file_type == 'application/pdf':
                    progress_text.text("Processing PDF...")
                    self.show_processing_animation("📄 Analyzing document content")
                    temp_file_path = os.path.join(self.temp_dir, uploaded_file.name)
                    with open(temp_file_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    doc_content = self.extract_text_from_pdf(temp_file_path)
                    image_paths = self.extract_images_from_pdf(temp_file_path)
                    progress_bar.progress(30)

                elif file_type == 'video/mp4':
                    progress_text.text("Processing video...")
                    self.show_processing_animation("🎥 Analyzing video content")
                    video_content = self.process_video_content(uploaded_file)
                    if video_content is None:
                        st.error("Video processing failed")
                        return
                    image_paths = video_content.get('image_paths', [])
                    progress_bar.progress(30)

                # Generate and process script
                if doc_content or video_content:  # Use 'or', as either can now provide content
                    progress_text.text("Generating script...")
                    self.show_processing_animation("✍️ Creating natural narration")
                    script = self.enhance_script_generation(doc_content, video_content)
                    progress_bar.progress(60)

                    if script:
                        progress_text.text("Creating audio...")
                        self.show_processing_animation("🎤 Generating voice narration")
                        audio_files = self.create_enhanced_audio(script)
                        progress_bar.progress(80)

                        if audio_files:
                            progress_text.text("Creating final video...")
                            self.show_processing_animation("🎬 Assembling final video")
                            video_path = self.create_video(image_paths, audio_files)
                            progress_bar.progress(100)
                            progress_text.empty()

                            if video_path and os.path.exists(video_path):
                                st.success("🎉 Your demo video is ready!")

                                # Display video and download options
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.video(video_path)
                                with col2:
                                    with open(video_path, "rb") as file:
                                        st.download_button(
                                            label="⬇️ Download Video",
                                            data=file,
                                            file_name="adoptly_demo.mp4",
                                            mime="video/mp4"
                                        )

                                    if st.button("📝 View Script"):
                                        st.markdown("### Generated Script")
                                        st.markdown(script)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

if __name__ == "__main__":
    app = EnhancedAdoptlyDemoCreator()
    app.main()
