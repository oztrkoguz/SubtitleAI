import yt_dlp
import whisper
import os
import tempfile
from moviepy.editor import VideoFileClip, VideoClip
import subprocess
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap

def detect_platform(url):
    """Detect platform type from URL"""
    if 'tiktok.com' in url or 'vm.tiktok.com' in url:
        return 'tiktok'
    elif 'youtube.com' in url or 'youtu.be' in url:
        return 'youtube'
    else:
        return 'unknown'

def download_video_and_audio(url):
    """Download video and audio file from YouTube/TikTok"""
    platform = detect_platform(url)
    print(f"Downloading video... (Platform: {platform.upper()})")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Format options based on platform
    if platform == 'tiktok':
        format_options = [
            # Simple formats for TikTok
            'best[ext=mp4]',
            'best',
            'worst[ext=mp4]',
            'worst'
        ]
    else:  # YouTube and others
        format_options = [
            # Audio and video together - best quality
            'best[ext=mp4][height<=720]',
            'best[ext=mp4][height<=480]', 
            'best[ext=mp4][height<=360]',
            # Audio with lowest video quality
            'worst[ext=mp4]+bestaudio/best[ext=mp4]',
            # For m3u8 formats
            '231+233/231+234',  # 480p + audio
            '230+233/230+234',  # 360p + audio  
            '229+233/229+234',  # 240p + audio
            # Last resort - any format
            'best/worst'
        ]
    
    video_path = None
    
    for format_code in format_options:
        try:
            print(f"Trying format '{format_code}'...")
            
            # Platform-specific settings
            if platform == 'tiktok':
                opts = {
                    'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                    'format': format_code,
                    'noplaylist': True,
                    'no_warnings': False,
                    'extractaudio': False,
                    # TikTok specific settings
                    'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
                    }
                }
            else:  # YouTube and others
                opts = {
                    'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                    'format': format_code,
                    'noplaylist': True,
                    'no_warnings': False,
                    'extractaudio': False,
                    # For merge process
                    'merge_output_format': 'mp4',
                    # Add cookie and user agent
                    'cookiefile': None,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
            # Find downloaded file
            video_files = [f for f in os.listdir(temp_dir) if f.endswith(('.mp4', '.webm', '.mkv'))]
            
            if video_files:
                video_path = os.path.join(temp_dir, video_files[0])
                print(f"✅ Format '{format_code}' successful!")
                print(f"Downloaded file: {video_files[0]}")
                break
            else:
                print(f"❌ Format '{format_code}' did not create file")
                
        except Exception as e:
            print(f"❌ Format '{format_code}' failed: {str(e)[:100]}...")
            continue
    
    if not video_path:
        raise Exception("No format worked. YouTube policies may have changed.")
    
    # Extract audio with MoviePy
    print("Extracting audio file...")
    try:
        video_clip = VideoFileClip(video_path)
        audio_path = os.path.join(temp_dir, 'audio.wav')
        video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video_clip.close()
        print(f"✅ Audio file extracted: audio.wav")
    except Exception as e:
        print(f"❌ Audio extraction error: {e}")
        raise
    
    # Copy video and audio files to working directory
    import shutil
    
    # Copy video file
    video_filename = os.path.basename(video_path)
    final_video_path = os.path.join(os.getcwd(), f"indirilen_video_{video_filename}")
    shutil.copy2(video_path, final_video_path)
    print(f"✅ Video copied to working directory: {final_video_path}")
    
    # Copy audio file
    final_audio_path = os.path.join(os.getcwd(), "indirilen_ses.wav")
    shutil.copy2(audio_path, final_audio_path)
    print(f"✅ Audio file copied to working directory: {final_audio_path}")
    
    return final_video_path, final_audio_path, temp_dir

def transcribe_audio_to_srt(audio_path, language='tr'):
    """Convert audio file to text with Whisper and create SRT"""
    print("Extracting text with Whisper...")
    
    try:
        # Load Whisper model (base model is fast and accurate enough)
        model = whisper.load_model("base")
        print("✅ Whisper model loaded")
        
        # Transcribe audio file
        print("Starting transcription... (This may take some time)")
        result = model.transcribe(audio_path, language=language)  # Language code specified here
        print("✅ Transcription completed")
        
        # Save SRT file to working directory
        srt_path = os.path.join(os.getcwd(), 'subtitles.srt')
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"]):
                start = seconds_to_srt_time(segment["start"])
                end = seconds_to_srt_time(segment["end"])
                text = segment["text"].strip()
                
                # Check for empty text
                if not text:
                    continue
                
                # Write in SRT format
                f.write(f"{i + 1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        
        # Check SRT file content
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if len(content.strip()) == 0:
                raise Exception("SRT file was created empty")
        
        print(f"SRT file content ({len(content)} characters):")
        print(content[:200] + "..." if len(content) > 200 else content)
        
        print(f"✅ SRT file created: {srt_path}")
        return srt_path
        
    except Exception as e:
        print(f"❌ Whisper error: {e}")
        raise

def seconds_to_srt_time(seconds):
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def add_subtitles_to_video(video_path, srt_path, output_path, subtitle_config=None):
    """Save video with subtitles using FFmpeg"""
    print("Adding subtitles to video...")
    
    # Check file paths
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    if not os.path.exists(srt_path):
        print(f"❌ SRT file not found: {srt_path}")
        return False
    
    # Check font file existence
    font_family_path = subtitle_config.get('font_family') if subtitle_config else None
    if font_family_path:
        print(f"🔍 Checking font file: {font_family_path}")
        if os.path.exists(font_family_path):
            print(f"✅ Font file found: {os.path.basename(font_family_path)}")
        else:
            print(f"❌ Font file not found: {font_family_path}")
    
    # Default settings
    if subtitle_config is None:
        subtitle_config = {
            'font_size': 24,
            'font_color': '#FFFFFF',
            'text_position': 'bottom',
            'font_family': None
        }
    
    print(f"🎨 Subtitle settings: {subtitle_config}")
    
    # Try PIL method
    print("🎨 PIL-based subtitle creation")
    pil_success = create_subtitled_video_pil(video_path, srt_path, output_path, subtitle_config)
    
    if pil_success:
        print("✅ PIL method successful!")
        return True
    
    print("⚠️ PIL method failed, switching to FFmpeg methods...")
    
    # FFmpeg backup method
    try:
        print("🔄 Trying FFmpeg simple subtitles method...")
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f"subtitles={srt_path}",
            '-c:a', 'copy',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check if output file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ Subtitled video created (FFmpeg): {output_path}")
            return True
        else:
            print("❌ FFmpeg did not create file")
            return False
                
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed:")
        print(f"Return code: {e.returncode}")
        if e.stderr:
            print(f"Stderr: {e.stderr[:200]}...")
        return False
        
    except FileNotFoundError:
        print("❌ FFmpeg not found. Make sure FFmpeg is installed and in PATH.")
        return False

def cleanup_temp_files(temp_dir):
    """Clean up temporary files"""
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print("✅ Temporary files cleaned up")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

def parse_srt_file(srt_path):
    """Parse SRT file and return subtitle list"""
    subtitles = []
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        blocks = content.split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Index (not used)
                # Time range
                time_line = lines[1]
                # Text (can be multiple lines)
                text = '\n'.join(lines[2:])
                
                # Parse times
                start_str, end_str = time_line.split(' --> ')
                start_time = srt_time_to_seconds(start_str)
                end_time = srt_time_to_seconds(end_str)
                
                subtitles.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })
        
        print(f"✅ SRT parsed: {len(subtitles)} subtitle segments")
        return subtitles
        
    except Exception as e:
        print(f"❌ SRT parsing error: {e}")
        return []

def srt_time_to_seconds(time_str):
    """Convert SRT time format to seconds (00:01:30,500 -> 90.5)"""
    time_part, ms_part = time_str.split(',')
    h, m, s = map(int, time_part.split(':'))
    ms = int(ms_part)
    
    total_seconds = h * 3600 + m * 60 + s + ms / 1000.0
    return total_seconds

def create_text_frame(text, frame_size, config):
    """Create subtitle frame with PIL"""
    img = Image.new('RGBA', frame_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        if config.get('font_family') and config['font_family'] != 'Default':
            font = ImageFont.truetype(config['font_family'], config['font_size'])
        else:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"⚠️ Font loading failed: {e}, using default")
        font = ImageFont.load_default()
    
    # Wrap text
    wrapped_text = textwrap.fill(text, width=50)
    
    # Calculate text dimensions
    bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate text position
    x = (frame_size[0] - text_width) // 2
    if config['text_position'] == 'bottom':
        y = frame_size[1] - text_height - 50
    elif config['text_position'] == 'top':
        y = 50
    else:  # middle
        y = (frame_size[1] - text_height) // 2
    
    # Draw colored text (use hex color)
    draw.text(
        (x, y), 
        wrapped_text, 
        font=font, 
        fill=config['font_color']  # Hex color string (#FFFF00 etc.)
    )
    
    return np.array(img)

def create_subtitled_video_pil(video_path, srt_path, output_path, subtitle_config):
    """Create subtitled video with PIL"""
    print("🎨 Creating subtitled video with PIL...")
    
    try:
        # Parse SRT file
        subtitles = parse_srt_file(srt_path)
        if not subtitles:
            return False
        
        # Load video
        video = VideoFileClip(video_path)
        frame_size = (int(video.w), int(video.h))
        
        print(f"📺 Video size: {frame_size}")
        print(f"⏱️ Video duration: {video.duration:.2f} seconds")
        print(f"📝 Number of subtitles: {len(subtitles)}")
        
        def make_frame(t):
            frame = video.get_frame(t)
            
            # Which subtitle is active at current time?
            current_subtitle = None
            for subtitle in subtitles:
                if subtitle['start'] <= t <= subtitle['end']:
                    current_subtitle = subtitle
                    break
            
            if current_subtitle:
                # Create subtitle frame
                text_frame = create_text_frame(
                    current_subtitle['text'], 
                    frame_size,
                    subtitle_config
                )
                
                # Alpha blending
                text_frame_rgb = text_frame[..., :3]
                text_frame_alpha = text_frame[..., 3:] / 255.0
                
                frame = frame * (1.0 - text_frame_alpha) + \
                        text_frame_rgb * text_frame_alpha
            
            return frame
        
        # Create new video
        final_video = VideoClip(make_frame, duration=video.duration)
        final_video = final_video.set_audio(video.audio)
        
        # Save video
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps,
            verbose=False,
            logger=None
        )
        
        print(f"✅ PIL subtitled video saved: {output_path}")
        
        # Cleanup
        video.close()
        final_video.close()
        
        return True
        
    except Exception as e:
        print(f"❌ PIL video creation error: {e}")
        return False

def create_safe_filename(original_name):
    """Create safe filename (clean Turkish characters and special characters)"""
    import re
    import unicodedata
    
    # Normalize unicode characters
    normalized = unicodedata.normalize('NFKD', original_name)
    
    # Replace Turkish characters
    char_map = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
    }
    
    for turkish_char, english_char in char_map.items():
        normalized = normalized.replace(turkish_char, english_char)
    
    # Keep only alphanumeric characters, hyphens and underscores
    safe_name = re.sub(r'[^\w\-_.]', '_', normalized)
    
    # Convert multiple underscores to single
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Remove leading and trailing underscores
    safe_name = safe_name.strip('_')
    
    # Shorten if too long
    if len(safe_name) > 100:
        name_part, ext = os.path.splitext(safe_name)
        safe_name = name_part[:90] + ext
    
    return safe_name

def process_video_with_srt(video_url, subtitle_config=None, language='tr'):
    """
    Create SRT subtitled video from video URL (for app.py)
    
    Args:
        video_url (str): YouTube or TikTok URL
        subtitle_config (dict): Subtitle settings
        language (str): Language code for Whisper ('tr', 'en', etc.)
    
    Returns:
        tuple: (success, output_video_path, srt_path, error_message)
    """
    temp_dir = None
    try:
        print(f"🎥 Starting SRT subtitle process...")
        print(f"📹 Video URL: {video_url}")
        print(f"🌍 Language: {language}")
        print(f"🎨 Subtitle settings: {subtitle_config}")
        
        # 1. Download video and audio
        print("📥 Downloading video...")
        video_path, audio_path, temp_dir = download_video_and_audio(video_url)
        
        # 2. Rename video file with safe name
        original_basename = os.path.basename(video_path)
        safe_basename = create_safe_filename(original_basename)
        safe_video_path = os.path.join(os.path.dirname(video_path), safe_basename)
        
        # Copy video file with safe name
        import shutil
        shutil.copy2(video_path, safe_video_path)
        print(f"📁 Video copied with safe name: {safe_basename}")
        
        # 3. Create subtitles with Whisper
        print("🎙️ Converting speech to text...")
        srt_path = transcribe_audio_to_srt(audio_path, language=language)
        
        # 4. Determine output filename
        output_filename = f"srt_subtitled_{safe_basename}"
        output_path = os.path.join(os.getcwd(), output_filename)
        
        print(f"📁 Output file: {output_filename}")
        
        # 5. Add subtitles to video (use safe video file)
        print("🎬 Adding subtitles to video...")
        success = add_subtitles_to_video(safe_video_path, srt_path, output_path, subtitle_config)
        
        if success:
            # Clean up temporary files
            cleanup_temp_files(temp_dir)
            return True, output_path, srt_path, None
        else:
            # Clean up temporary files
            cleanup_temp_files(temp_dir)
            return False, None, srt_path, "Subtitle addition failed"
            
    except Exception as e:
        error_msg = f"SRT process error: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Clean up temporary files
        if temp_dir:
            cleanup_temp_files(temp_dir)
            
        return False, None, None, error_msg