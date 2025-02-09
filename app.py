import streamlit as st
import os
from PIL import Image
import pypdfium2 as pdfium
# pandas and pathlib aren't directly used, but are good practice for file handling
# so I'm leaving them.  Remove if you're *sure* you don't need them.
import pandas as pd
from pathlib import Path  # Not strictly needed
import time # Time functions included. Can be deleted without damage.
from moviepy.editor import *
import openai
from gtts import gTTS
import tempfile
import base64
import pytesseract
import cv2
import numpy as np
from dotenv import load_dotenv # Not actively used
import io # not used actively.
import re # RE operations
from scipy.io.wavfile import write
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
        """Generate script, including intro of AI and the main product."""
        try:
            intro = "Hello! I'm Adoptly Demo Creator. I've analyzed the provided content. I will guide you through a quick demonstration of the key features and benefits."

            key_points = self.extract_key_points(content)

            if video_content:
                duration = video_content['timing_info']['duration']
                segments = self.plan_video_segments(duration, key_points)
                video_prompt = f"""
                {intro}

                Here is your product demo, targeted for {duration:.2f} seconds:

                Key points to highlight are:
                {key_points}

                Video segment timing details:
                {segments}

                Adhere to these points:
                1. Focus on core value propositions at the begining
                2. keep sentences easy to listen, like conversational
                3. Smooth and clear transition from topics to topics.
                4. Segment timings needs to followed as told.
                5. Prioritize and speak benefits of product more often.

                Please create the presentation keeping conversation approach in mind. Include some brief and minimal timestamps like \[MM:SS] marking transition periods
                """
            else: # for PDFs:
                words = len(content.split())
                estimated_duration = (words / 150) * 60
                segments = self.plan_video_segments(estimated_duration, key_points)
                video_prompt = f"""
                {intro}

                Here is your product demo for about {estimated_duration:.0f} seconds:

                Core information to discuss are:
                {key_points}

                Please, use this for your timing segment plans:
                {segments}

                Please follow guidelines below
                1. Focus on what user gain and what you'll benefit from
                2. Create natural flow with sentence, keep conversations flow running
                3. Create transitions for the topic and create them smooth.
                4. segment durations should match to give an optimal pacing.
                5. Use conversational ton of words in speaking
                I want the video should be conversational.
                Please mark and follow these time stamps (like \[MM:SS])
                """
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Expert presenter creating natural, engaging demo narration."},
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
        """Extract key points from content"""
        try:
            prompt = f"""
            Identify and pull the key points. Prioritize these when summarizing
            1. Value propositions that matter
            2. Primary benefits
            3. Primary functions and features
            4. Important metrics that describe it

            Content:
            {content[:3000]} # limiting for tokens
            """
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Good at selecting and summarize key info from product demo"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"Error extracting key points: {str(e)}")
            return content[:1000] # default fallback


    def plan_video_segments(self, duration, key_points):
        """Plan video segments."""
        try:
            prompt = f"""
            Plan the content to build demo. The time limit you have is {duration:.0f} seconds and focus is:
            {key_points}
            Here is timing breakdown:
            1. Give great pace.
            2. create clean transitions for viewer to understand and think.
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Creating the pacing plan and timing of video"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"Error planning segments: {str(e)}")
            return f"Create an plan evenly in {duration:.0f} second period."


    def post_process_script(self, script):
        """Process to make good output format."""
        script = re.sub(r'\n{3,}', '\n\n', script) # cleanup multiple \n
        script = re.sub(r'\[(\d+):(\d+)\]', lambda m: f'[{int(m.group(1)):02d}:{int(m.group(2)):02d}]', script)
        script = script.replace('AI:','').replace('Narrator:','').strip() #removing artifacts from processing steps

        return script

    def create_enhanced_audio(self, script):
        """Generate better quality audio files without stutter."""
        try:
            segments = self.split_script_into_segments(script)
            audio_files = []

            for segment in segments:
                clean_text = self.clean_text_for_tts(segment)

                if clean_text:
                    tts = gTTS(text=clean_text, lang='en', slow=False)
                    audio_path = os.path.join(self.temp_dir, f'audio_{len(audio_files)}.mp3')
                    tts.save(audio_path)
                    self.add_pause_to_audio(audio_path) # small pauses
                    audio_files.append(audio_path)
            return audio_files
        except Exception as e:
            st.error(f"Error creating audio: {str(e)}")
            return None


    def split_script_into_segments(self, script):
        """Break script into shorter segment to reduce stutter and break from sentences to increase efficiency."""
        sentences = re.split(r'(?<=[.!?])\s+', script.strip())
        return [s for s in sentences if s.strip()]

    def clean_text_for_tts(self, text):
        """Make good text for TTS audio"""
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip() # replace long strings with simple string
        text = text.replace('.', '. ').replace('!', '! ').replace('?', '? ')

        return text

    def add_pause_to_audio(self, audio_path):
        """Make small pause in between segment."""
        try:
            audio = AudioFileClip(audio_path)
            pause = AudioFileClip(self.create_pause(0.3))
            final_audio = concatenate_audioclips([audio, pause])
            final_audio.write_audiofile(audio_path)
        except Exception as e:
            st.warning(f"Error adding pause to audio: {str(e)}")

    def create_pause(self, duration):
        """Making audio pause by the duration."""
        pause_path = os.path.join(self.temp_dir, 'pause.mp3')
        silence = np.zeros(int(duration * 44100))  # 44100 Hz
        write(pause_path, 44100, silence.astype(np.float32))
        return pause_path

    def create_video(self, image_paths, audio_files, background_path=None):
        """Create video, use the audio duration, and sync audio. Fix aspect ratio issue."""
        try:
            if not image_paths or not audio_files:
                st.error("No pictures or audio.")
                return None

            clips = []
            total_duration = 0
            # Set current_audio_time by adding the the current segment

            #Calculate audio for full content by concatenating all audio clips
            for audio_file in audio_files:
                audio_clip = AudioFileClip(audio_file)
                total_duration += audio_clip.duration
                audio_clip.close()

            # Time duration on pictures to balance each picture. This make no error like division by zero
            time_per_image = total_duration / len(image_paths) if len(image_paths) > 0 else 0
            audio_clip_index = 0 # pointer that indicates which audio clip needs to get in play
            current_audio_time = 0

            for i, image_path in enumerate(image_paths):
                image_clip = ImageClip(image_path)
                #Determine the duration that based on duration that coming.

                if audio_clip_index < len(audio_files):
                    audio_clip = AudioFileClip(audio_files[audio_clip_index])
                    image_duration = audio_clip.duration # here make a good and dynamic picture
                    audio_clip.close()

                else:  # default case if issue exist and audio and video dont matches
                    image_duration = time_per_image

                image_clip = image_clip.set_duration(image_duration) # set dynamic value
                # Here the size will get resize the picture, keeping ratio to the frame in a great view!
                target_width = 1280
                target_height = 720

                image_clip = image_clip.resize(lambda t: (target_width, target_height) if image_clip.w > target_width or image_clip.h > target_height else (image_clip.w, image_clip.h))
                image_clip = image_clip.fadein(0.5).fadeout(0.5) # transition. fadein & fadeout for images
                image_clip = image_clip.set_pos("center") # keep this to keep a nice central point
                clips.append(image_clip)

                current_audio_time += image_duration  # update the current clip duration here!
                #audio_clip = AudioFileClip(audio_files[audio_clip_index])
                if current_audio_time >= sum(AudioFileClip(af).duration for af in audio_files[:audio_clip_index+1]): # sync
                    audio_clip_index +=1 # increase number of play

            final_video = concatenate_videoclips(clips, method="compose")
            audio_clips = [AudioFileClip(af) for af in audio_files]
            final_audio = concatenate_audioclips(audio_clips)

            final_video = final_video.set_audio(final_audio)
            # Make length of audio clips same as of videos length!

            if final_video.duration > final_audio.duration: # Make them aligned by making them same lengh
                final_video = final_video.subclip(0,final_audio.duration)

            elif final_audio.duration > final_video.duration: # trim the time frame for sound and add the sounds accordingly!

                 difference_duration = final_audio.duration - final_video.duration
                 pause_audio_path = self.create_pause(difference_duration)
                 pause_audio_clip = AudioFileClip(pause_audio_path)
                 list_for_concatinate = [final_audio, pause_audio_clip] # adding pause for finishing up the sound
                 final_audio = concatenate_audioclips(list_for_concatinate)
                 final_video = final_video.set_audio(final_audio)

            output_path = os.path.join(self.temp_dir, 'final_video.mp4')
            final_video.write_videofile(output_path, fps=24, codec='libx264',
                                        preset="medium", bitrate="5000k",
                                        audio_codec='aac') # save a nice video

            # remove every clip which has some sort of issue
            final_video.close()
            final_audio.close()
            for clip in audio_clips:
                clip.close() # this closes them completely
            for clip in clips:
                clip.close()

            return output_path # giving out the video output in directory


        except Exception as e:
            st.error(f"Error on create video : {str(e)}")
            return None

    def show_processing_animation(self, message):
        """Show processing animation with updates"""
        st.markdown(f"""
            <div class="processing-animation">
                <div class="loader"></div>
                <p>{message}</p>
            </div>
            """, unsafe_allow_html=True)


    def main(self):
        """Main and complete user interface that does it's functions and create a working UI."""
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%); border-radius: 15px; margin-bottom: 2rem;">
                <h1 style="color: white;">🎥 Adoptly Demo Creator</h1>
                <p style="color: white; font-size: 1.2rem;">Create engaging product demos with natural AI narration</p>
            </div>
            """, unsafe_allow_html=True)
        if not self.setup_api_key():
            return # do nothing on error if setup key
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
                    voice_speed = st.slider("Narration Speed", 0.8, 1.2, 1.0, 0.1)  # slider to change speed
                    enable_transitions = st.checkbox("Enable Visual Transitions", value=True)  # use animation while demo!
                    high_quality = st.checkbox("High Quality Processing", value=True)  # High quaility to process for high range output


                doc_content = "" # content related variables initialization
                temp_file_path = None
                image_paths = []
                video_content = None
                audio_files = []
                script = None

                # Upload and proccessing
                file_type = uploaded_file.type
                progress_text = st.empty()
                progress_bar = st.progress(0)

                if file_type == 'application/pdf':
                    progress_text.text("Processing PDF...") # to see and keep eye to watch processing stage
                    self.show_processing_animation("📄 Analyzing document content") # for loader show in pdf
                    temp_file_path = os.path.join(self.temp_dir, uploaded_file.name)
                    with open(temp_file_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    doc_content = self.extract_text_from_pdf(temp_file_path) # give doc contents!
                    image_paths = self.extract_images_from_pdf(temp_file_path) # extract
                    progress_bar.progress(30) # move process


                elif file_type == 'video/mp4':
                    progress_text.text("Processing video...") # change processing bar
                    self.show_processing_animation("🎥 Analyzing video content")
                    video_content = self.process_video_content(uploaded_file) # call function to make an working output.
                    if video_content is None: # give if issues with function!
                        st.error("Video processing failed")
                        return
                    image_paths = video_content.get('image_paths', [])
                    progress_bar.progress(30)

                #Script related steps

                if doc_content or video_content: # call create output as required based on content
                    progress_text.text("Generating script...") # text bar to let to see processing
                    self.show_processing_animation("✍️ Creating natural narration") # create AI base animation loader

                    script = self.enhance_script_generation(doc_content, video_content) # creating video here

                    progress_bar.progress(60) # increasing

                    if script:
                        progress_text.text("Creating audio...") # making text
                        self.show_processing_animation("🎤 Generating voice narration")

                        audio_files = self.create_enhanced_audio(script)
                        progress_bar.progress(80)

                        if audio_files: # last processing here!
                            progress_text.text("Creating final video...")
                            self.show_processing_animation("🎬 Assembling final video")
                            video_path = self.create_video(image_paths, audio_files)
                            progress_bar.progress(100)
                            progress_text.empty() # remove the text

                            if video_path and os.path.exists(video_path): # for file check as well!
                                st.success("🎉 Your demo video is ready!")
                                #Video with options here
                                col1, col2 = st.columns([2, 1])

                                with col1:
                                    st.video(video_path)
                                with col2: # second column has the files
                                    with open(video_path, "rb") as file: # here the videos file should present
                                        st.download_button(
                                            label="⬇️ Download Video",
                                            data=file,
                                            file_name="adoptly_demo.mp4", # for naming file,
                                            mime="video/mp4" # for getting format right.
                                        )
                                    if st.button("📝 View Script"):
                                        st.markdown("### Generated Script")
                                        st.markdown(script) # here get scrit to look it!

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.error(traceback.format_exc()) # show error logs.

if __name__ == "__main__":
    app = EnhancedAdoptlyDemoCreator()
    app.main()
