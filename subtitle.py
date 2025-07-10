from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import VideoFileClip, VideoClip
import textwrap
import os

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

def create_subtitled_video(video_path, scene_descriptions, scene_times, subtitle_config=None, output_path=None):
    """Create subtitled video with user preferences"""
    try:
        print("Creating subtitled video...")
        print("\n=== Inside create_subtitled_video ===")
        print(f"Number of descriptions: {len(scene_descriptions)}")
        print(f"First description: {scene_descriptions[0]}")
        print("================================\n")
        
        if subtitle_config is None:
            subtitle_config = {
                'font_size': 40,
                'font_color': '#FFFF00',
                'text_position': 'bottom'
            }
        
        print("Creating subtitled video...")
        
        video = VideoFileClip(video_path)
        frame_size = (int(video.w), int(video.h))
        
        def make_frame(t):
            frame = video.get_frame(t)
            
            current_scene_idx = None
            for i, start_time in enumerate(scene_times):
                if i < len(scene_times) - 1:
                    if start_time <= t < scene_times[i + 1]:
                        current_scene_idx = i
                        break
                else:
                    if start_time <= t:
                        current_scene_idx = i
                        break
            
            if current_scene_idx is not None:
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
            output_path = f"{base_name}_with_subtitles.mp4"
        
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps
        )
        
        print(f"Subtitled video saved: {output_path}")
        
        video.close()
        final_video.close()
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error adding subtitles: {str(e)}") 