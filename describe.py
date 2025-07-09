import torch
from PIL import Image
from decord import VideoReader, cpu
from scenedetect import detect, ContentDetector
import yt_dlp
import os
import tempfile
import base64
import re
import ollama
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

def encode_image(image_path):
    """Resmi base64 formatına çevirir"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_overall_impression(text):
    """Overall Impression kısmının açıklamasını çıkarır"""
    
    # YÖNTEM 1: Overall Impression'dan sonraki ilk paragrafı al
    if "**Overall Impression:**" in text:
        parts = text.split("**Overall Impression:**", 1)
        if len(parts) > 1:
            after_impression = parts[1].strip()
            # Bir sonraki ** işaretine kadar olan kısmı al
            next_section = after_impression.split("**")[0].strip()
            return next_section
    
    # YÖNTEM 2: Regex ile daha hassas çıkarma
    pattern = r'\*\*Overall Impression:\*\*\s*(.*?)(?=\*\*|\n\n|\Z)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return "Overall Impression bulunamadı"

# Çeviri için ChatOllama kullanacağız, model yüklemeye gerek yok



def download_youtube_video(url):
    """Download YouTube video and return temp file path"""
    try:
        print("Downloading video...")
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = os.path.join(tempfile.gettempdir(), f"{info['id']}.mp4")
            print(f"Video downloaded successfully: {video_path}")
            return video_path
            
    except Exception as e:
        raise Exception(f"Video download error: {str(e)}")

def detect_scenes(video_path):
    """Detect scene transitions in video"""
    try:
        print("Detecting scenes...")
        scenes = detect(video_path, ContentDetector())
        scene_times = [scene[0].get_seconds() for scene in scenes]
        print(f"Number of scenes detected: {len(scene_times)}")
        return scene_times
    except Exception as e:
        raise Exception(f"Scene detection error: {str(e)}")

def encode_video(video_path):
    """Extract one frame from each scene"""
    try:
        print(f"Processing video: {video_path}")
        scene_times = detect_scenes(video_path)
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        frames = []
        frame_times = []
        
        for i in range(len(scene_times)):
            start_time = scene_times[i]
            if i < len(scene_times) - 1:
                end_time = scene_times[i + 1]
            else:
                end_time = total_frames / fps
            
            middle_time = (start_time + end_time) / 2
            frame_idx = int(middle_time * fps)
            
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1
                
            try:
                frame = vr.get_batch([frame_idx]).asnumpy()[0]
                frames.append(Image.fromarray(frame.astype('uint8')))
                frame_times.append(start_time)
                print(f"Scene {i+1}/{len(scene_times)} processed - Time: {start_time:.2f}s")
            except Exception as e:
                print(f"Error processing scene {i+1}: {str(e)}")
        
        return frames, scene_times
        
    except Exception as e:
        raise Exception(f"Video processing error: {str(e)}")

def calculate_optimal_description_length(scene_duration):
    """Calculate optimal description length based on scene duration"""
    # Average speech rate: 2.5 words/second
    WORDS_PER_SECOND = 2.5
    max_words = int(scene_duration * WORDS_PER_SECOND)
    # Minimum 3, maximum 15 words
    return max(3, min(15, max_words))

def generate_frame_descriptions(frames, scene_times):
    """Generate English descriptions for each scene using Ollama"""
    try:
        print("Describing scenes with Ollama...")
        print(f"Total scenes: {len(frames)}")
        
        descriptions = []
        temp_dir = tempfile.mkdtemp()
        
        for i, frame in enumerate(frames):
            try:
                # Frame'i geçici dosya olarak kaydet
                temp_image_path = os.path.join(temp_dir, f"frame_{i}.jpg")
                frame.save(temp_image_path)
                
                # Base64'e çevir
                base64_image = encode_image(temp_image_path)
                
                # Sahne süresine göre optimal uzunluk hesapla
                scene_duration = scene_times[i+1] - scene_times[i] if i < len(scene_times)-1 else 5
                optimal_length = calculate_optimal_description_length(scene_duration)
                
                # Ollama ile görüntü analizi
                response = ollama.chat(
                    model='gemma3:4b',  # Vision modeli kullan
                    messages=[
                        {
                            'role': 'user',
                            'content': f'Describe the main action in this scene in {optimal_length} words or less. Focus on key actions and subjects.',
                            'images': [base64_image]
                        }
                    ]
                )
                
                # Yanıtı işle
                full_description = response['message']['content']
                
                # Overall Impression varsa çıkar, yoksa tam metni kullan
                scene_desc = extract_overall_impression(full_description)
                if scene_desc == "Overall Impression bulunamadı":
                    scene_desc = full_description
                
                # Kelime sayısını sınırla
                words = scene_desc.split()[:optimal_length]
                scene_desc = ' '.join(words)
                if not scene_desc.endswith('.'):
                    scene_desc = scene_desc.rstrip(',') + '.'
                
                descriptions.append(scene_desc)
                print(f"Scene {i+1}/{len(frames)} described: {scene_desc}")
                
                # Geçici dosyayı sil
                os.remove(temp_image_path)
                
            except Exception as e:
                print(f"Error describing scene {i+1}: {str(e)}")
                descriptions.append(f"Scene {i+1}")
                continue
        
        # Geçici dizini temizle
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
        return descriptions
        
    except Exception as e:
        print(f"Description error: {str(e)}")
        return [f"Scene {i+1}" for i in range(len(frames))]

def translate_and_enhance_with_ollama(descriptions, target_lang):
    """Translate descriptions using Ollama"""
    try:
        print(f"Translating descriptions to {target_lang}...")
        
        # If target language is English, no translation needed
        if target_lang == 'en':
            return descriptions
        
        # Dil kodlarını tam isme çevir
        lang_names = {
            'en': 'English',
            'tr': 'Turkish'
        }
        
        source_lang = lang_names.get('en', 'English')
        target_lang_name = lang_names.get(target_lang, 'Turkish')
        
        print(f"Starting translation from English to {target_lang_name}")
        print(f"Using model: qwen3:8b")
        
        enhanced_descriptions = []
        for i, desc in enumerate(descriptions):
            print(f"\nTranslating ({i+1}/{len(descriptions)})")
            print(f"Original: {desc}")
            
            try:
                # Ollama ile çeviri
                response = ollama.chat(
                    model='gemma3:4b',
                    messages=[
                        {
                            'role': 'user',
                            'content': f'You are a professional translator. Translate the following {source_lang} text to {target_lang_name}. Only provide the translation, no explanations: "{desc}"'
                        }
                    ],
                    options={
                        'temperature': 0.1
                    }
                )
                
                enhanced = response['message']['content'].strip().replace('\n', ' ')
                
                enhanced_descriptions.append(enhanced)
                print(f"Translated: {enhanced}")
                
            except Exception as e:
                print(f"Error translating description {i+1}: {str(e)}")
                enhanced_descriptions.append(desc)  # Hata durumunda orijinali kullan
                
        return enhanced_descriptions
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return descriptions

def summarize_with_ollama(scene_descriptions, lang='en'):
    """Create summary in selected language"""
    try:
        print("Generating summary...")
        
        llm = OllamaLLM(model="mistral")
        
        # Summary templates for English and Turkish
        summary_templates = {
            'tr': """
            Aşağıda bir videodan farklı sahnelerin açıklamaları bulunmaktadır.
            Bu açıklamaları kullanarak videonun tutarlı ve anlamlı bir özetini oluşturun.

            Sahneler:
            {scene_descriptions}

            Lütfen:
            1. Gereksiz bilgileri birleştirin
            2. Önemli ayrıntıları koruyun
            3. Kronolojik sırayı koruyun
            4. Akıcı ve bağlantılı cümleler kullanın
            5. Ana olaylara ve eylemlere odaklanın
            6. Doğal olarak akan bir anlatı oluşturun

            Videonun özünü yakalayan kapsamlı bir özet yazın:
            """,
            
            'en': """
            Below are descriptions of different scenes from a video. 
            Create a coherent and meaningful summary of the video using these descriptions.

            Scenes:
            {scene_descriptions}

            Please:
            1. Combine redundant information
            2. Preserve important details
            3. Maintain chronological order
            4. Use fluid and connected sentences
            5. Focus on the main events and actions
            6. Create a narrative that flows naturally

            Write a comprehensive summary that captures the essence of the video:
                """
        }
        
        template = summary_templates.get(lang, summary_templates['en'])
        
        # Sahne açıklamalarını madde işaretleriyle birleştir
        scenes = "\n".join([f"- {desc}" for desc in scene_descriptions])
        
        # Özet oluştur
        prompt = template.format(scene_descriptions=scenes)
        final_summary = llm.invoke(prompt)
        
        # Gereksiz boşlukları ve fazla satırları temizle
        final_summary = final_summary.strip()
        
        return final_summary
        
    except Exception as e:
        print(f"Summary generation error: {str(e)}")
        return ' '.join(scene_descriptions) 