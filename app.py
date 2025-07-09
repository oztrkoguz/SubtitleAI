import gradio as gr
import os
from describe import (
    download_youtube_video, 
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

# Available fonts
SUBTITLE_FONTS = {
    'Arial': 'arial.ttf',
    'Times New Roman': 'times.ttf',
    'Verdana': 'verdana.ttf',
    'Courier New': 'cour.ttf',
    'Comic Sans': 'comic.ttf',
    'Impact': 'impact.ttf',
    'Default': None
}

def process_video_with_lang(video_input, video_url, selected_lang, font_size, font_color, text_position, font_family):
    """Process video and generate summary"""
    try:
        # 1. Process video
        if video_url:
            video_path = download_youtube_video(video_url)
        elif isinstance(video_input, str) and (video_input.startswith(('http://', 'https://', 'www.', 'youtube.com', 'youtu.be'))):
            video_path = download_youtube_video(video_input)
        else:
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
                # Video input
                video_input = gr.Video(label="Upload Video")
                video_url = gr.Textbox(label="Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=example")
                
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
                video_url,
                language,
                font_size,
                font_color,
                text_position,
                font_family
            ],
            outputs=[output_video, summary_text]
        )
        
        gr.Markdown("""
        ### How to Use:
        1. Upload a video or enter a YouTube URL
        2. Select description and voice language
        3. Customize subtitle appearance:
           - Adjust font size
           - Choose font color
           - Set text position
        4. Click 'Process Video'
        5. Wait for the processed video and summary
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)