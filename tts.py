from TTS.api import TTS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import numpy as np
import os
import tempfile
import shutil
import soundfile as sf
from scipy import signal

def ensure_temp_dir():
    """Create directory for temporary files"""
    temp_dir = os.path.join(os.getcwd(), 'temp_audio')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

def cleanup_temp_dir(temp_dir):
    """Clean up temporary files and directory"""
    try:
        shutil.rmtree(temp_dir)
        print("Geçici dosyalar temizlendi")
    except Exception as e:
        print(f"Geçici dosyaları temizlerken hata: {str(e)}")

def calculate_speech_speed(text, scene_duration):
    """Calculate speech speed based on scene duration"""
    WORDS_PER_SECOND = 2.5
    word_count = len(text.split())
    normal_duration = word_count / WORDS_PER_SECOND
    target_duration = scene_duration * 0.9
    
    if normal_duration > target_duration:
        return normal_duration / target_duration
    else:
        return 1.0

def speed_up_audio_file(input_file, output_file, speed_factor):
    """Speed up audio file"""
    if speed_factor <= 1.0:
        return input_file
        
    # Read audio file
    data, samplerate = sf.read(input_file)
    
    # Calculate new length
    new_length = int(len(data) / speed_factor)
    
    # Resample audio data
    if len(data.shape) == 2:  # Stereo
        resampled = np.zeros((new_length, 2))
        for channel in range(2):
            resampled[:, channel] = signal.resample(data[:, channel], new_length)
    else:  # Mono
        resampled = signal.resample(data, new_length)
    
    # Save new audio file
    sf.write(output_file, resampled, samplerate)
    
    return output_file

def create_video_with_tts(video_path, scene_descriptions, scene_times, output_path=None, lang='en'):
    """Create video with text-to-speech narration"""
    try:
        print("Creating voiced video...")
        temp_dir = ensure_temp_dir()
        
        # TTS models for English and Turkish only
        model_map = {
            'tr': "tts_models/tr/common-voice/glow-tts",
            'en': "tts_models/en/ljspeech/glow-tts"
        }
        
        model_name = model_map.get(lang, model_map['en'])
        tts = TTS(model_name=model_name, progress_bar=False)
        print(f'Loaded TTS model: {model_name}')
        
        # Load main video
        video = VideoFileClip(video_path).set_audio(None)
        audio_clips = []
        
        for i, description in enumerate(scene_descriptions):
            try:
                # Calculate scene duration
                current_time = scene_times[i]
                next_time = scene_times[i + 1] if i < len(scene_times) - 1 else video.duration
                scene_duration = next_time - current_time
                
                # Create audio file
                temp_audio = os.path.join(temp_dir, f'scene_{i}.wav')
                
                # Generate TTS audio
                tts.tts_to_file(
                    text=description,
                    file_path=temp_audio
                )
                
                # Calculate and adjust speech speed
                speed = calculate_speech_speed(description, scene_duration)
                if speed > 1.0:
                    fast_audio = os.path.join(temp_dir, f'scene_{i}_fast.wav')
                    temp_audio = speed_up_audio_file(temp_audio, fast_audio, speed)
                
                # Create and position audio clip
                audio_clip = AudioFileClip(temp_audio)
                audio_clip = audio_clip.set_start(current_time)
                
                # Fit audio duration to scene
                if audio_clip.duration > scene_duration:
                    audio_clip = audio_clip.set_duration(scene_duration)
                
                audio_clips.append(audio_clip)
                print(f"Scene {i+1} audio created (duration: {scene_duration:.1f}s, speed: {speed:.1f}x)")
                
            except Exception as e:
                print(f"Error creating audio for scene {i+1}: {str(e)}")
                continue
        
        if not audio_clips:
            raise Exception("No audio clips were created!")
        
        # Combine audio and add to video
        final_audio = CompositeAudioClip(audio_clips)
        final_video = video.set_audio(final_audio)
        
        if output_path is None:
            base_name = os.path.splitext(video_path)[0]
            output_path = f"{base_name}_with_tts.mp4"
        
        # Save video
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps
        )
        
        print(f"Video with TTS saved: {output_path}")
        
        # Clean up resources
        video.close()
        final_video.close()
        for clip in audio_clips:
            clip.close()
        
        cleanup_temp_dir(temp_dir)
        return output_path
        
    except Exception as e:
        if temp_dir:
            cleanup_temp_dir(temp_dir)
        raise Exception(f"Error creating TTS video: {str(e)}") 