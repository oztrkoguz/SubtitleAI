import yt_dlp
import whisper
import os
import tempfile
from moviepy.editor import VideoFileClip, VideoClip
import subprocess
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import re
from datetime import datetime, timedelta
import random

class AdvancedSubtitleProcessor:
    
    def __init__(self):
        self.user_preferences = {}
        self.colors = {
            'Yellow': '#FFFF00',
            'White': '#FFFFFF', 
            'Red': '#FF0000',
            'Blue': '#0080FF',
            'Green': '#00FF00',
            'Purple': '#FF00FF',
            'Orange': '#FF8000',
            'Pink': '#FF80C0',
            'Gold': '#FFD700',
            'Silver': '#C0C0C0',
            'Neon Green': '#39FF14',
            'Neon Blue': '#1B03A3',
            'Turquoise': '#40E0D0',
            'Lavender': '#E6E6FA',
            'Lime': '#32CD32'
        }
        
        self.effects = {
            'fade': 'Smooth Fade',
            'slide_up': 'Slide Up',
            'slide_down': 'Slide Down', 
            'slide_left': 'Slide Left',
            'slide_right': 'Slide Right',
            'zoom': 'Zoom In/Out',
            'zoom_in': 'Zoom In',
            'zoom_out': 'Zoom Out',
            'glow': 'Glow Effect',
            'shake': 'Shake',
            'rotate_cw': 'Rotate Clockwise',
            'rotate_ccw': 'Rotate Counter-Clockwise', 
            'wave': 'Wave Motion',
            'pulse': 'Pulse',
            'flip': 'Flip',
            'spiral': 'Spiral',
            'elastic': 'Elastic',
            'bounce': 'Bounce',
            'mixed': 'Mixed (Random)',
            'none': 'No Effect'
        }
        
        self.positions = {
            'bottom': 'Bottom Center',
            'bottom_left': 'Bottom Left',
            'bottom_right': 'Bottom Right',
            'middle': 'Middle Center',
            'top': 'Top Center',
            'top_left': 'Top Left',
            'top_right': 'Top Right'
        }
    
    def set_preferences_from_gradio(self, font_size, font_color, text_position, font_family, 
                                   effect_type='fade', outline_size=3, shadow_size=2, opacity=1.0):
        """Gradio arayüzünden gelen ayarları işle"""
        self.user_preferences = {
            'color': {'name': font_color, 'hex': font_color if font_color.startswith('#') else self.colors.get(font_color, '#FFFF00')},
            'effect': {'name': self.effects.get(effect_type, 'Yumuşak Geçiş'), 'code': effect_type},
            'position': {'name': self.positions.get(text_position, 'Alt Orta'), 'alignment': text_position},
            'font_size': font_size,
            'font_name': font_family or 'Arial Black',
            'outline': {'name': f'Kalın ({outline_size}px)', 'value': str(outline_size)},
            'shadow': {'name': f'Gölge ({shadow_size}px)', 'value': str(shadow_size)},
            'opacity': {'name': f'Şeffaflık ({opacity})', 'value': opacity}
        }
        
        print(f"✅ Gelişmiş altyazı ayarları uygulandı:")
        print(f"   Renk: {self.user_preferences['color']['name']}")
        print(f"   Efekt: {self.user_preferences['effect']['name']}")
        print(f"   Konum: {self.user_preferences['position']['name']}")
        print(f"   Font: {self.user_preferences['font_name']} ({font_size}px)")
        print(f"   Çerçeve: {outline_size}px, Gölge: {shadow_size}px")
    
    def get_enhanced_text_properties(self):
        """Gelişmiş metin özelliklerini döndür"""
        if not self.user_preferences:
            return {
                'color': '#FFFF00',
                'font_size': 28,
                'font_name': 'Arial Black',
                'outline_size': 3,
                'shadow_size': 2,
                'position': 'bottom',
                'effect': 'fade',
                'opacity': 1.0
            }
        
        color_hex = self.user_preferences['color']['hex']
        if not color_hex.startswith('#'):
            color_hex = self.colors.get(color_hex, '#FFFF00')
        
        return {
            'color': color_hex,
            'font_size': self.user_preferences['font_size'],
            'font_name': self.user_preferences['font_name'],
            'outline_size': int(self.user_preferences['outline']['value']),
            'shadow_size': int(self.user_preferences['shadow']['value']),
            'position': self.user_preferences['position']['alignment'],
            'effect': self.user_preferences['effect']['code'],
            'opacity': float(self.user_preferences['opacity']['value'])
        }
    
    def apply_text_effects(self, text, time_position=0.0):
        """Metne zaman bazlı efektler uygula"""
        if not self.user_preferences:
            return text, {}
        
        effect_type = self.user_preferences['effect']['code']
        effect_params = {}
        
        # Efekt parametrelerini hesapla
        if effect_type == 'fade':
            effect_params['alpha'] = min(time_position * 3, 1.0)
        elif effect_type == 'slide_up':
            effect_params['offset_y'] = int(20 * (1 - min(time_position * 2, 1)))
        elif effect_type == 'slide_down':
            effect_params['offset_y'] = int(-20 * (1 - min(time_position * 2, 1)))
        elif effect_type == 'slide_left':
            effect_params['offset_x'] = int(30 * (1 - min(time_position * 2, 1)))
        elif effect_type == 'slide_right':
            effect_params['offset_x'] = int(-30 * (1 - min(time_position * 2, 1)))
        elif effect_type in ['zoom', 'zoom_in']:
            effect_params['scale'] = 0.8 + 0.2 * min(time_position * 2, 1)
        elif effect_type == 'zoom_out':
            effect_params['scale'] = 1.2 - 0.2 * min(time_position * 2, 1)
        elif effect_type == 'pulse':
            pulse_factor = 1 + 0.1 * abs(np.sin(time_position * 6))
            effect_params['scale'] = pulse_factor
        elif effect_type == 'wave':
            wave_factor = 0.05 * np.sin(time_position * 8)
            effect_params['wave_offset'] = int(wave_factor * 10)
        elif effect_type == 'shake':
            if time_position < 0.5:
                shake_x = random.randint(-2, 2)
                shake_y = random.randint(-2, 2)
                effect_params['offset_x'] = shake_x
                effect_params['offset_y'] = shake_y
        elif effect_type == 'rotate_cw':
            effect_params['rotation'] = time_position * 360
        elif effect_type == 'rotate_ccw':
            effect_params['rotation'] = -time_position * 360
        elif effect_type == 'mixed':
            # Karışık efektler
            effect_params['alpha'] = min(time_position * 2, 1.0)
            effect_params['scale'] = 1 + 0.05 * np.sin(time_position * 4)
            effect_params['offset_y'] = int(5 * np.sin(time_position * 3))
        
        return text, effect_params

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

def add_subtitles_to_video(video_path, srt_path, output_path, subtitle_config=None, advanced_processor=None):
    """Save video with subtitles using enhanced PIL method"""
    print("Adding enhanced subtitles to video...")
    
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
    
    if advanced_processor:
        print(f"✨ Enhanced subtitle processor enabled")
    
    # Try enhanced PIL method
    print("🎨 Enhanced PIL-based subtitle creation")
    pil_success = create_subtitled_video_pil(video_path, srt_path, output_path, subtitle_config, advanced_processor)
    
    if pil_success:
        print("✅ Enhanced PIL method successful!")
        return True
    
    print("⚠️ Enhanced PIL method failed, switching to FFmpeg fallback...")
    
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
            print(f"✅ Subtitled video created (FFmpeg fallback): {output_path}")
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

def create_enhanced_text_frame(text, frame_size, config, advanced_processor=None, time_position=0.0):
    """Create enhanced subtitle frame with PIL - Gelişmiş efektlerle"""
    img = Image.new('RGBA', frame_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Gelişmiş ayarları al
    if advanced_processor and advanced_processor.user_preferences:
        enhanced_props = advanced_processor.get_enhanced_text_properties()
        font_size = enhanced_props['font_size']
        font_color = enhanced_props['color']
        outline_size = enhanced_props['outline_size']
        shadow_size = enhanced_props['shadow_size']
        position = enhanced_props['position']
        opacity = enhanced_props['opacity']
        
        # Efekt uygula
        processed_text, effect_params = advanced_processor.apply_text_effects(text, time_position)
    else:
        # Varsayılan ayarlar
        font_size = config.get('font_size', 28)
        font_color = config.get('font_color', '#FFFF00')
        outline_size = 3
        shadow_size = 2
        position = config.get('text_position', 'bottom')
        opacity = 1.0
        effect_params = {}
    
    # Font yükle
    try:
        if config.get('font_family') and config['font_family'] != 'Default':
            base_font_size = int(font_size * effect_params.get('scale', 1.0))
            font = ImageFont.truetype(config['font_family'], base_font_size)
        else:
            # Varsayılan bold font dene
            base_font_size = int(font_size * effect_params.get('scale', 1.0))
            try:
                font = ImageFont.truetype('arial.ttf', base_font_size)
            except:
                try:
                    font = ImageFont.truetype('Arial.ttf', base_font_size)
                except:
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
    
    # Calculate base text position
    x = (frame_size[0] - text_width) // 2
    if position in ['bottom', 'bottom_left', 'bottom_right']:
        y = frame_size[1] - text_height - 50
    elif position in ['top', 'top_left', 'top_right']:
        y = 50
    else:  # middle
        y = (frame_size[1] - text_height) // 2
    
    # Pozisyon ayarlamaları
    if 'left' in position:
        x = 50
    elif 'right' in position:
        x = frame_size[0] - text_width - 50
    
    # Efekt offset'lerini uygula
    x += effect_params.get('offset_x', 0)
    y += effect_params.get('offset_y', 0)
    y += effect_params.get('wave_offset', 0)
    
    # Alpha hesapla
    alpha_multiplier = effect_params.get('alpha', opacity)
    
    # Gölge çiz (eğer varsa)
    if shadow_size > 0:
        shadow_color = '#000000'
        shadow_alpha = int(128 * alpha_multiplier)
        for sx in range(-shadow_size, shadow_size + 1):
            for sy in range(-shadow_size, shadow_size + 1):
                if sx != 0 or sy != 0:
                    try:
                        draw.text(
                            (x + sx, y + sy),
                            wrapped_text,
                            font=font,
                            fill=shadow_color + f'{shadow_alpha:02x}'
                        )
                    except:
                        draw.text(
                            (x + sx, y + sy),
                            wrapped_text,
                            font=font,
                            fill=shadow_color
                        )
    
    # Çerçeve çiz (outline)
    if outline_size > 0:
        outline_color = '#000000'
        outline_alpha = int(255 * alpha_multiplier)
        for ox in range(-outline_size, outline_size + 1):
            for oy in range(-outline_size, outline_size + 1):
                if ox != 0 or oy != 0:
                    try:
                        draw.text(
                            (x + ox, y + oy),
                            wrapped_text,
                            font=font,
                            fill=outline_color + f'{outline_alpha:02x}'
                        )
                    except:
                        draw.text(
                            (x + ox, y + oy),
                            wrapped_text,
                            font=font,
                            fill=outline_color
                        )
    
    # Ana metni çiz
    main_alpha = int(255 * alpha_multiplier)
    try:
        if len(font_color) == 7:  # #RRGGBB formatı
            font_color_with_alpha = font_color + f'{main_alpha:02x}'
        else:
            font_color_with_alpha = font_color
        
        draw.text(
            (x, y),
            wrapped_text,
            font=font,
            fill=font_color_with_alpha
        )
    except:
        # Alpha desteği yoksa normal renk kullan
        draw.text(
            (x, y),
            wrapped_text,
            font=font,
            fill=font_color
        )
    
    return np.array(img)

def create_text_frame(text, frame_size, config):
    """Eski create_text_frame fonksiyonu - geriye uyumluluk için"""
    return create_enhanced_text_frame(text, frame_size, config)

def create_subtitled_video_pil(video_path, srt_path, output_path, subtitle_config, advanced_processor=None):
    """Create subtitled video with PIL - Gelişmiş efektlerle"""
    print("🎨 Creating enhanced subtitled video with PIL...")
    
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
        
        if advanced_processor and advanced_processor.user_preferences:
            print(f"🎨 Enhanced subtitle effects enabled")
            enhanced_props = advanced_processor.get_enhanced_text_properties()
            print(f"   Color: {enhanced_props['color']}")
            print(f"   Effect: {enhanced_props['effect']}")
            print(f"   Font size: {enhanced_props['font_size']}")
            print(f"   Position: {enhanced_props['position']}")
        
        def make_frame(t):
            frame = video.get_frame(t)
            
            # Which subtitle is active at current time?
            current_subtitle = None
            subtitle_start_time = None
            for subtitle in subtitles:
                if subtitle['start'] <= t <= subtitle['end']:
                    current_subtitle = subtitle
                    subtitle_start_time = subtitle['start']
                    break
            
            if current_subtitle:
                # Calculate time position within subtitle (0.0 to 1.0)
                subtitle_duration = current_subtitle['end'] - subtitle_start_time
                time_in_subtitle = t - subtitle_start_time
                time_position = time_in_subtitle / subtitle_duration if subtitle_duration > 0 else 0
                
                # Create enhanced subtitle frame
                text_frame = create_enhanced_text_frame(
                    current_subtitle['text'], 
                    frame_size,
                    subtitle_config,
                    advanced_processor,
                    time_position
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
        
        print(f"✅ Enhanced PIL subtitled video saved: {output_path}")
        
        # Cleanup
        video.close()
        final_video.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced PIL video creation error: {e}")
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

def process_video_with_srt(video_url, subtitle_config=None, language='tr', advanced_config=None):
    """
    Create enhanced SRT subtitled video from video URL (for app.py)
    
    Args:
        video_url (str): YouTube or TikTok URL
        subtitle_config (dict): Basic subtitle settings
        language (str): Language code for Whisper ('tr', 'en', etc.)
        advanced_config (dict): Advanced subtitle effects configuration
    
    Returns:
        tuple: (success, output_video_path, srt_path, error_message)
    """
    temp_dir = None
    try:
        print(f"🎥 Starting enhanced SRT subtitle process...")
        print(f"📹 Video URL: {video_url}")
        print(f"🌍 Language: {language}")
        print(f"🎨 Subtitle settings: {subtitle_config}")
        print(f"✨ Advanced config: {advanced_config}")
        
        # Gelişmiş işlemci oluştur
        advanced_processor = None
        if advanced_config:
            advanced_processor = AdvancedSubtitleProcessor()
            advanced_processor.set_preferences_from_gradio(
                font_size=advanced_config.get('font_size', subtitle_config.get('font_size', 28)),
                font_color=advanced_config.get('font_color', subtitle_config.get('font_color', '#FFFF00')),
                text_position=advanced_config.get('text_position', subtitle_config.get('text_position', 'bottom')),
                font_family=advanced_config.get('font_family', subtitle_config.get('font_family', 'Arial Black')),
                effect_type=advanced_config.get('effect_type', 'fade'),
                outline_size=advanced_config.get('outline_size', 3),
                shadow_size=advanced_config.get('shadow_size', 2),
                opacity=advanced_config.get('opacity', 1.0)
            )
        
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
        prefix = "enhanced_srt" if advanced_processor else "srt_subtitled"
        output_filename = f"{prefix}_{safe_basename}"
        output_path = os.path.join(os.getcwd(), output_filename)
        
        print(f"📁 Output file: {output_filename}")
        
        # 5. Add enhanced subtitles to video
        print("🎬 Adding enhanced subtitles to video...")
        success = add_subtitles_to_video(safe_video_path, srt_path, output_path, subtitle_config, advanced_processor)
        
        if success:
            # Clean up temporary files
            cleanup_temp_files(temp_dir)
            effect_info = f" with {advanced_config.get('effect_type', 'fade')} effects" if advanced_processor else ""
            return True, output_path, srt_path, f"Enhanced subtitles added successfully{effect_info}!"
        else:
            # Clean up temporary files
            cleanup_temp_files(temp_dir)
            return False, None, srt_path, "Enhanced subtitle addition failed"
            
    except Exception as e:
        error_msg = f"Enhanced SRT process error: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Clean up temporary files
        if temp_dir:
            cleanup_temp_files(temp_dir)
            
        return False, None, None, error_msg
