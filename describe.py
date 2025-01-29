import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
from decord import VideoReader, cpu
from scenedetect import detect, ContentDetector
import yt_dlp
import os
import tempfile
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Load models
print("Loading models...")
model = AutoModel.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

# Load M2M100 model and tokenizer
print("Loading translation model...")
model_name = "facebook/m2m100_418M"
m2m_tokenizer = M2M100Tokenizer.from_pretrained(model_name)
m2m_model = M2M100ForConditionalGeneration.from_pretrained(model_name)

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

def generate_frame_descriptions(frames, tokenizer, model, scene_times):
    """Generate English descriptions for each scene"""
    try:
        print("Describing scenes...")
        print(f"Total scenes: {len(frames)}")
        
        # Use English prompt only
        prompt_template = "Describe the main action in this scene in {} words or less."
        
        descriptions = []
        for i, frame in enumerate(frames):
            scene_duration = scene_times[i+1] - scene_times[i] if i < len(scene_times)-1 else 5
            optimal_length = calculate_optimal_description_length(scene_duration)
            
            prompt = prompt_template.format(optimal_length)
            msgs = [{'role': 'user', 'content': [frame] + [prompt]}]
            
            params = {
                "use_image_id": False,
                "max_slice_nums": 2
            }
            
            description = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                **params
            )
            
            words = description.split()[:optimal_length]
            scene_desc = ' '.join(words)
            if not scene_desc.endswith('.'):
                scene_desc = scene_desc.rstrip(',') + '.'
                
            descriptions.append(scene_desc)
            print(f"Scene {i+1}/{len(frames)} described")
        
        return descriptions
        
    except Exception as e:
        print(f"Description error: {str(e)}")
        return [f"Scene {i+1}" for i in range(len(frames))]

def translate_and_enhance_with_m2m(descriptions, target_lang):
    """Translate descriptions using M2M100"""
    try:
        print(f"Translating descriptions to {target_lang}...")
        
        # Translate only for Turkish
        if target_lang != 'tr':
            return descriptions
        
        enhanced_descriptions = []
        for i, desc in enumerate(descriptions):
            print(f"\nTranslating ({i+1}/{len(descriptions)})")
            print(f"Original: {desc}")
            
            # Translate with M2M100
            m2m_tokenizer.src_lang = "en"
            encoded = m2m_tokenizer(desc, return_tensors="pt")
            generated_tokens = m2m_model.generate(
                **encoded,
                forced_bos_token_id=m2m_tokenizer.get_lang_id("tr")
            )
            enhanced = m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            enhanced = enhanced.strip().replace('\n', ' ')
            enhanced_descriptions.append(enhanced)
            print(f"Translated: {enhanced}")
            
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