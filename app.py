import gradio as gr
import os
from describe import (
    download_youtube_video, 
    download_video_from_url,
    encode_video, 
    generate_frame_descriptions, 
    summarize_with_ollama,
    translate_and_enhance_with_ollama
)
from subtitle import create_subtitled_video
from tts import create_video_with_tts

# Supported languages
LANGUAGES = {
    'English': 'en',
    'Turkish': 'tr'
}

# Predefined colors for subtitles
SUBTITLE_COLORS = {
    'Yellow': '#FFFF00',
    'White': '#FFFFFF',
    'Red': '#FF0000',
    'Green': '#00FF00',
    'Blue': '#0000FF',
    'Orange': '#FFA500',
    'Pink': '#FFC0CB'
}

# Available fonts - dinamik olarak fonts klasöründen yükle
def get_available_fonts():
    fonts_dir = os.path.join(os.getcwd(), 'fonts')
    fonts = {'Default': None}  # Varsayılan font
    
    # Fonts klasörü yoksa oluştur
    if not os.path.exists(fonts_dir):
        os.makedirs(fonts_dir)
        print(f"Fonts klasörü oluşturuldu: {fonts_dir}")
        print("TTF font dosyalarınızı bu klasöre ekleyebilirsiniz.")
    
    if os.path.exists(fonts_dir):
        font_count = 0
        for font_file in os.listdir(fonts_dir):
            if font_file.lower().endswith('.ttf'):
                font_name = os.path.splitext(font_file)[0]
                font_path = os.path.join(fonts_dir, font_file)
                fonts[font_name] = font_path
                font_count += 1
        
        print(f"Fonts klasöründen {font_count} adet font yüklendi.")
        if font_count == 0:
            print("Fonts klasörü boş. TTF dosyalarınızı ekleyin.")
                
    return fonts

SUBTITLE_FONTS = get_available_fonts()

def process_video_with_lang(video_input, youtube_url, tiktok_url, selected_lang, font_size, font_color, text_position, font_family):
    """Process video and generate summary"""
    try:
        # 1. Process video - URL priority: TikTok > YouTube > Upload
        if tiktok_url and tiktok_url.strip():
            print("Processing TikTok URL...")
            video_path = download_video_from_url(tiktok_url.strip())
        elif youtube_url and youtube_url.strip():
            print("Processing YouTube URL...")
            video_path = download_video_from_url(youtube_url.strip())
        elif isinstance(video_input, str) and (video_input.startswith(('http://', 'https://'))):
            print("Processing URL from video input...")
            video_path = download_video_from_url(video_input)
        else:
            print("Processing uploaded video...")
            video_path = video_input.name if hasattr(video_input, 'name') else video_input
            
        frames, scene_times = encode_video(video_path)
        
        # 2. Generate English descriptions
        original_descriptions = generate_frame_descriptions(frames, scene_times)
        
        # 3. Language check and translation
        lang_code = LANGUAGES[selected_lang]
        print(f"[APP] Çeviri fonksiyonu çağrılıyor - target_lang: {lang_code}")
        # Translate if not English
        if lang_code != 'en':
            print("Translating descriptions...")
            scene_descriptions = translate_and_enhance_with_ollama(original_descriptions, lang_code)
            print("\nFirst scene description:")
            print(f"Original (EN): {original_descriptions[0]}")
            print(f"Enhanced (TR): {scene_descriptions[0]}")
        else:
            scene_descriptions = original_descriptions
        
        # 4. Create subtitled video
        subtitle_config = {
            'font_size': font_size,
            'font_color': SUBTITLE_COLORS[font_color],
            'text_position': text_position,
            'font_family': SUBTITLE_FONTS[font_family]
        }
        
        # 5. Create TTS video
        final_video = create_video_with_tts(
            create_subtitled_video(
                video_path=video_path, 
                scene_descriptions=scene_descriptions, 
                scene_times=scene_times,
                subtitle_config=subtitle_config
            ), 
            scene_descriptions,  
            scene_times, 
            lang=lang_code
        )
        
        # 6. Clean up temp files
        try:
            os.remove(video_path)
        except:
            pass
            
        # 7. Generate summary in selected language
        summary = summarize_with_ollama(scene_descriptions, lang=lang_code)
        
        return final_video, summary
        
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

# Gradio interface
def create_interface():
    with gr.Blocks(title="AI-SubtitleSum") as interface:
        gr.Markdown("# 🎥 SubtitleAI ")
        
        with gr.Row():
            with gr.Column():
                # Video input options
                gr.Markdown("### 📹 Video Input Options")
                video_input = gr.Video(label="📁 Upload Video File")
                
                with gr.Row():
                    youtube_url = gr.Textbox(
                        label="🎬 YouTube URL", 
                        placeholder="https://www.youtube.com/watch?v=example",
                        scale=1
                    )
                    tiktok_url = gr.Textbox(
                        label="📱 TikTok URL", 
                        placeholder="https://www.tiktok.com/@user/video/123456789",
                        scale=1
                    )
                
                # Language selection
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="Description and Voice Language"
                )
                
                # Subtitle customization
                gr.Markdown("### Subtitle Settings")
                with gr.Row():
                    with gr.Column():
                        font_size = gr.Slider(
                            minimum=20,
                            maximum=80,
                            value=40,
                            step=2,
                            label="Font Size"
                        )
                        
                        font_color = gr.Dropdown(
                            choices=list(SUBTITLE_COLORS.keys()),
                            value="Yellow",
                            label="Font Color"
                        )
                        
                        font_family = gr.Dropdown(
                            choices=list(SUBTITLE_FONTS.keys()),
                            value="Arial",
                            label="Font Family"
                        )
                        
                        text_position = gr.Radio(
                            choices=["bottom", "top", "middle"],
                            value="bottom",
                            label="Text Position"
                        )
                
                # Process button
                process_btn = gr.Button("Process Video", variant="primary")
            
            with gr.Column():
                # Output video
                output_video = gr.Video(label="Processed Video")
                
                # Summary text
                summary_text = gr.Textbox(
                    label="Video Summary",
                    lines=5,
                    interactive=False
                )
        
        # Process button click event
        process_btn.click(
            fn=process_video_with_lang,
            inputs=[
                video_input, 
                youtube_url,
                tiktok_url,
                language,
                font_size,
                font_color,
                text_position,
                font_family
            ],
            outputs=[output_video, summary_text]
        )
        
        gr.Markdown("""
        ### 📖 How to Use:
        1. **Choose video input method:**
           - 📁 Upload a video file OR
           - 🎬 Enter YouTube URL OR  
           - 📱 Enter TikTok URL
        2. **Select language** for descriptions and voice
        3. **Customize subtitles:**
           - Font size, color, position, family
        4. **Click 'Process Video'**
        5. **Wait** for AI processing and final video
        
        ⚠️ **Note:** TikTok videos are optimized for short content analysis
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)