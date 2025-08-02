from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import VideoFileClip, VideoClip
import textwrap
import os
import random
from srt_subtitle import AdvancedSubtitleProcessor, create_enhanced_text_frame

def create_text_frame(text, frame_size, config):
    """Create subtitle frame with user preferences"""
    img = Image.new('RGBA', frame_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        if config['font_family']:
            font = ImageFont.truetype(config['font_family'], config['font_size'])
        else:
            font = ImageFont.load_default()
    except:
        print(f"Warning: Could not load font {config['font_family']}, using default")
        font = ImageFont.load_default()
    
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
    
    # Draw colored text using hex color
    draw.text(
        (x, y), 
        wrapped_text, 
        font=font, 
        fill=config['font_color']  # Hex color string (#FFFF00 etc.)
    )
    
    return np.array(img)

def create_subtitled_video(video_path, scene_descriptions, scene_times, subtitle_config=None, output_path=None, advanced_config=None):
    """Create subtitled video with enhanced effects support"""
    try:
        print(f"Creating {'enhanced ' if advanced_config else ''}subtitled video...")
        print("\n=== Inside create_subtitled_video ===")
        print(f"Number of descriptions: {len(scene_descriptions)}")
        print(f"First description: {scene_descriptions[0]}")
        if advanced_config:
            print(f"Advanced effects enabled: {advanced_config}")
        print("================================\n")
        
        if subtitle_config is None:
            subtitle_config = {
                'font_size': 40,
                'font_color': '#FFFF00',
                'text_position': 'bottom'
            }
        
        # Gelişmiş işlemci oluştur (eğer gerekirse)
        advanced_processor = None
        if advanced_config:
            advanced_processor = AdvancedSubtitleProcessor()
            advanced_processor.set_preferences_from_gradio(
                font_size=advanced_config.get('font_size', subtitle_config.get('font_size', 40)),
                font_color=advanced_config.get('font_color', subtitle_config.get('font_color', '#FFFF00')),
                text_position=advanced_config.get('text_position', subtitle_config.get('text_position', 'bottom')),
                font_family=advanced_config.get('font_family', subtitle_config.get('font_family', 'Arial Black')),
                effect_type=advanced_config.get('effect_type', 'fade'),
                outline_size=advanced_config.get('outline_size', 3),
                shadow_size=advanced_config.get('shadow_size', 2),
                opacity=advanced_config.get('opacity', 1.0)
            )
            print("✨ Advanced subtitle processor initialized")
        
        print("Creating subtitled video...")
        
        video = VideoFileClip(video_path)
        frame_size = (int(video.w), int(video.h))
        
        def make_frame(t):
            frame = video.get_frame(t)
            
            current_scene_idx = None
            scene_start_time = None
            for i, start_time in enumerate(scene_times):
                if i < len(scene_times) - 1:
                    if start_time <= t < scene_times[i + 1]:
                        current_scene_idx = i
                        scene_start_time = start_time
                        break
                else:
                    if start_time <= t:
                        current_scene_idx = i
                        scene_start_time = start_time
                        break
            
            if current_scene_idx is not None:
                # Zaman pozisyonunu hesapla (sahne içindeki geçen süre)
                time_in_scene = t - scene_start_time if scene_start_time is not None else 0
                
                # Sahne süresini hesapla
                if current_scene_idx < len(scene_times) - 1:
                    scene_duration = scene_times[current_scene_idx + 1] - scene_start_time
                else:
                    scene_duration = video.duration - scene_start_time
                
                # Zaman pozisyonu (0.0 - 1.0 arası)
                time_position = time_in_scene / scene_duration if scene_duration > 0 else 0
                
                # Gelişmiş veya basit metin frame oluştur
                if advanced_processor:
                    text_frame = create_enhanced_text_frame(
                        scene_descriptions[current_scene_idx], 
                        frame_size,
                        subtitle_config,
                        advanced_processor,
                        time_position
                    )
                else:
                    text_frame = create_text_frame(
                        scene_descriptions[current_scene_idx], 
                        frame_size,
                        subtitle_config
                    )
                
                text_frame_rgb = text_frame[..., :3]
                text_frame_alpha = text_frame[..., 3:] / 255.0
                
                frame = frame * (1.0 - text_frame_alpha) + \
                        text_frame_rgb * text_frame_alpha
            
            return frame
        
        final_video = VideoClip(make_frame, duration=video.duration)
        final_video = final_video.set_audio(video.audio)
        
        if output_path is None:
            base_name = os.path.splitext(video_path)[0]
            prefix = "enhanced" if advanced_processor else "subtitled"
            output_path = f"{base_name}_{prefix}.mp4"
        
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps
        )
        
        print(f"{'Enhanced ' if advanced_processor else ''}Subtitled video saved: {output_path}")
        if advanced_processor:
            enhanced_props = advanced_processor.get_enhanced_text_properties()
            print(f"✨ Applied effects: {enhanced_props['effect']} with {enhanced_props['color']} color")
        
        video.close()
        final_video.close()
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error adding subtitles: {str(e)}") 