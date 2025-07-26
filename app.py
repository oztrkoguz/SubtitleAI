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
from video_chat import YouTubeToText
from srt_subtitle import process_video_with_srt

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

# Global RAG sistemi
rag_system = None
current_transcript = ""

def initialize_rag_system():
    """RAG sistemini başlat"""
    global rag_system
    try:
        rag_system = YouTubeToText(
            enable_rag=True,
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("✅ RAG sistemi başlatıldı")
        return True
    except Exception as e:
        print(f"❌ RAG sistem hatası: {e}")
        return False

def setup_chatbot_only(video_input, youtube_url, tiktok_url, selected_lang):
    """Setup chatbot without video processing"""
    global rag_system, current_transcript
    
    # Determine video URL from inputs
    video_url = None
    if tiktok_url and tiktok_url.strip():
        video_url = tiktok_url.strip()
    elif youtube_url and youtube_url.strip():
        video_url = youtube_url.strip()
    elif isinstance(video_input, str) and (video_input.startswith(('http://', 'https://'))):
        video_url = video_input
    
    if not video_url:
        return "❌ Please enter a valid video URL (YouTube or TikTok)", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    if not rag_system:
        if not initialize_rag_system():
            return "❌ RAG system initialization failed", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    try:
        # Convert language to code
        lang_code = LANGUAGES[selected_lang]
        print(f"🤖 Setting up chatbot for video: {video_url}")
        print(f"🌍 Language: {selected_lang} ({lang_code})")
        
        result = rag_system.process_with_rag(video_url, method="whisper", language=lang_code)
        
        if result.get("transcript") and result.get("rag_ready"):
            current_transcript = result["transcript"]
            status = f"✅ Video Chatbot is ready! Language: {selected_lang}. You can ask questions below."
            return status, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        else:
            return "❌ Failed to create RAG system", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
    except Exception as e:
        return f"❌ RAG error: {str(e)}", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def process_video_with_lang(video_input, youtube_url, tiktok_url, selected_lang, font_size, font_color, text_position, font_family):
    """Process video and generate summary"""
    global current_transcript, rag_system
    
    try:
        # 1. Process video - URL priority: TikTok > YouTube > Upload
        video_url = None
        if tiktok_url and tiktok_url.strip():
            print("Processing TikTok URL...")
            video_path = download_video_from_url(tiktok_url.strip())
            video_url = tiktok_url.strip()
        elif youtube_url and youtube_url.strip():
            print("Processing YouTube URL...")
            video_path = download_video_from_url(youtube_url.strip())
            video_url = youtube_url.strip()
        elif isinstance(video_input, str) and (video_input.startswith(('http://', 'https://'))):
            print("Processing URL from video input...")
            video_path = download_video_from_url(video_input)
            video_url = video_input
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
        
        # 6. Generate summary in selected language
        summary = summarize_with_ollama(scene_descriptions, lang=lang_code)
        
        # 7. RAG sistemi için transkript oluştur (sadece URL'ler için)
        rag_status = "RAG system not available (URLs only)"
        chat_interface_visible = False
        
        if video_url and rag_system:
            try:
                print("🤖 Creating transcript for RAG system...")
                result = rag_system.process_with_rag(video_url, method="whisper", language=lang_code)
                
                if result.get("transcript") and result.get("rag_ready"):
                    current_transcript = result["transcript"]
                    rag_status = f"✅ Video chatbot is ready! Language: {selected_lang}. You can ask questions below."
                    chat_interface_visible = True
                else:
                    rag_status = "❌ RAG system creation failed"
                    
            except Exception as e:
                rag_status = f"❌ RAG error: {str(e)}"
        
        # 8. Clean up temp files
        try:
            os.remove(video_path)
        except:
            pass
            
        return final_video, summary, rag_status, gr.update(visible=chat_interface_visible), gr.update(visible=chat_interface_visible), gr.update(visible=chat_interface_visible)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return None, error_msg, "❌ RAG system unavailable due to error", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def ask_question_about_video(question, chat_history):
    """Ask questions about the video"""
    global rag_system, current_transcript
    
    if not question.strip():
        return chat_history, ""
    
    if not rag_system or not current_transcript:
        response = "❌ RAG system is not ready. Please process a video first or setup chatbot."
        chat_history.append([question, response])
        return chat_history, ""
    
    try:
        # Soru-cevap
        result = rag_system.ask_question(question)
        
        if result.get("error"):
            response = f"❌ Error: {result['error']}"
        else:
            response = result["answer"]
            
        chat_history.append([question, response])
        return chat_history, ""
        
    except Exception as e:
        response = f"❌ Q&A error: {str(e)}"
        chat_history.append([question, response])
        return chat_history, ""

def clear_chat():
    """Clear chat history"""
    return []

def process_srt_subtitles(video_input, youtube_url, tiktok_url, selected_lang, font_size, font_color, text_position, font_family):
    """SRT altyazı ile video işle - Whisper kullanarak"""
    try:
        # 1. Video URL'sini belirle - Öncelik: TikTok > YouTube > Upload
        video_url = None
        if tiktok_url and tiktok_url.strip():
            video_url = tiktok_url.strip()
        elif youtube_url and youtube_url.strip():
            video_url = youtube_url.strip()
        elif isinstance(video_input, str) and (video_input.startswith(('http://', 'https://'))):
            video_url = video_input
        
        if not video_url:
            return None, "❌ SRT altyazı oluşturma için video URL'si gerekli (YouTube veya TikTok). Lütfen geçerli bir URL girin."
        
        # 2. Dil kodunu çevir
        lang_code = LANGUAGES[selected_lang]
        
        # 3. Altyazı ayarlarını hazırla
        subtitle_config = {
            'font_size': font_size,
            'font_color': SUBTITLE_COLORS[font_color],
            'text_position': text_position,
            'font_family': SUBTITLE_FONTS[font_family]
        }
        
        print(f"🎬 SRT Altyazı işlemi başlıyor...")
        print(f"📹 URL: {video_url}")
        print(f"🌍 Dil: {selected_lang} ({lang_code})")
        print(f"🎨 Altyazı ayarları: {subtitle_config}")
        
        # 4. SRT ile video işle
        success, output_video_path, srt_path, error_message = process_video_with_srt(
            video_url=video_url,
            subtitle_config=subtitle_config,
            language=lang_code
        )
        
        if success:
            # Başarı mesajı oluştur
            success_msg = f"✅ SRT Altyazılı video başarıyla oluşturuldu!\n\n"
            success_msg += f"📹 Video: {os.path.basename(output_video_path)}\n"
            success_msg += f"📄 SRT dosyası: {os.path.basename(srt_path) if srt_path else 'N/A'}\n"
            success_msg += f"🌍 Dil: {selected_lang}\n"
            success_msg += f"🎨 Font boyutu: {font_size}, Renk: {font_color}, Pozisyon: {text_position}"
            
            return output_video_path, success_msg
        else:
            error_msg = f"❌ SRT altyazı oluşturma başarısız!\n\n"
            if error_message:
                error_msg += f"Hata: {error_message}\n\n"
            if srt_path:
                error_msg += f"📄 SRT dosyası oluşturuldu: {os.path.basename(srt_path)}\n"
                error_msg += "💡 SRT dosyasını video oynatıcınıza manuel olarak yükleyebilirsiniz."
            
            return None, error_msg
            
    except Exception as e:
        error_msg = f"❌ SRT işlemi sırasında beklenmeyen hata: {str(e)}"
        return None, error_msg

# Gradio interface
def create_interface():
    # RAG sistemini başlat
    initialize_rag_system()
    
    with gr.Blocks(
        title="AI-SubtitleSum", 
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .video-section { border: 2px solid #e1e5e9; border-radius: 10px; padding: 20px; margin: 10px 0; }
        .chat-section { border: 2px solid #d4edda; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f8f9fa; }
        .button-section { border: 2px solid #ffeaa7; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #fffbf0; }
        .status-box { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .success-status { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .error-status { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 🎥 SubtitleAI + Video Chatbot
            ### AI-powered video subtitle generation and intelligent chat system
            """, 
            elem_classes="main-container"
        )
        
        with gr.Row():
            # Left panel - Video processing
            with gr.Column(scale=1):
                with gr.Group(elem_classes="video-section"):
                    gr.Markdown("### 📹 Video Input")
                    
                    # Video input options
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
                        label="🌍 Description and Voice Language"
                    )
                
                with gr.Group(elem_classes="video-section"):
                    gr.Markdown("### 🎨 Subtitle Settings")
                    
                    with gr.Row():
                        with gr.Column():
                            font_size = gr.Slider(
                                minimum=20,
                                maximum=80,
                                value=40,
                                step=2,
                                label="📏 Font Size"
                            )
                            
                            font_color = gr.Dropdown(
                                choices=list(SUBTITLE_COLORS.keys()),
                                value="Yellow",
                                label="🎨 Font Color"
                            )
                            
                            font_family = gr.Dropdown(
                                choices=list(SUBTITLE_FONTS.keys()),
                                value="Default",
                                label="🔤 Font Family"
                            )
                            
                            text_position = gr.Radio(
                                choices=["bottom", "top", "middle"],
                                value="bottom",
                                label="📍 Text Position"
                            )
                
                # Action buttons
                with gr.Group(elem_classes="button-section"):
                    gr.Markdown("### 🚀 Choose Action")
                    gr.Markdown("*Use the same video input above for all options*")
                    
                    with gr.Row():
                        process_btn = gr.Button(
                            "🎬 Process Video\n(AI Descriptions + TTS)", 
                            variant="primary", 
                            size="lg",
                            scale=1
                        )
                        
                        srt_btn = gr.Button(
                            "📝 Generate SRT Subtitles\n(Whisper Transcription)", 
                            variant="primary", 
                            size="lg",
                            scale=1
                        )
                        
                        setup_chatbot_btn = gr.Button(
                            "🤖 Setup Chatbot Only\n(Quick Chat)", 
                            variant="secondary", 
                            size="lg",
                            scale=1
                        )
            
            # Right panel - Outputs and Chat
            with gr.Column(scale=1):
                with gr.Group(elem_classes="video-section"):
                    gr.Markdown("### 📺 Processed Video")
                    # Output video
                    output_video = gr.Video(label="Processed Video")
                    
                    # Summary text
                    summary_text = gr.Textbox(
                        label="📄 Video Summary",
                        lines=4,
                        interactive=False
                    )
                
                # RAG Status
                rag_status = gr.Markdown(
                    "ℹ️ Choose an action: Process Video or Setup Chatbot",
                    elem_classes="status-box"
                )
                
                # Chat Interface
                with gr.Group(elem_classes="chat-section", visible=False) as chat_section:
                    gr.Markdown("### 🤖 Video Chatbot")
                    gr.Markdown("*Ask any questions about the video content*")
                    
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=300,
                        show_label=True,
                        visible=False
                    )
                    
                    with gr.Row(visible=False) as chat_input_row:
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about the video...",
                            scale=4,
                            lines=1
                        )
                        ask_btn = gr.Button("📤 Send", scale=1)
                        clear_btn = gr.Button("🗑️ Clear", scale=1)
        
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
            outputs=[
                output_video, 
                summary_text, 
                rag_status,
                chat_section,
                chatbot,
                chat_input_row
            ]
        )
        
        # Setup chatbot only button click event
        setup_chatbot_btn.click(
            fn=setup_chatbot_only,
            inputs=[video_input, youtube_url, tiktok_url, language],
            outputs=[
                rag_status,
                chat_section,
                chatbot,
                chat_input_row
            ]
        )
        
        # SRT subtitle button click event
        srt_btn.click(
            fn=process_srt_subtitles,
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
            outputs=[
                output_video, 
                summary_text
            ]
        )
        
        # Chat events
        ask_btn.click(
            fn=ask_question_about_video,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input]
        )
        
        question_input.submit(
            fn=ask_question_about_video,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=chatbot
        )
        
        # Instructions
        gr.Markdown("""
        ### 📖 How to Use:
        
        #### 📹 Step 1 - Video Input:
        - **Upload a video file** OR **Enter YouTube/TikTok URL**
        - **Select language** for transcription and descriptions
        - **Customize subtitle settings** (font, color, position)
        
        #### 🚀 Step 2 - Choose Action:
        - **🎬 Process Video**: Creates AI-described subtitled video with TTS
        - **📝 Generate SRT Subtitles**: Creates accurate Whisper-based subtitles with custom styling
        - **🤖 Setup Chatbot Only**: Quick chatbot setup without video processing
        
        #### 💬 Step 3 - Video Chat:
        - After video processing or chatbot setup, **chatbot becomes active**
        - **Ask questions** about the video content
        - **AI analyzes video audio** and provides intelligent answers
        
        #### ⚠️ Important Notes:
        - **SRT subtitles and chatbot require URLs** (YouTube/TikTok) - not uploaded files
        - **TikTok videos** optimized for short content analysis
        - **AI Processing** includes descriptions + TTS + chatbot
        - **SRT Generation** uses Whisper for accurate transcription with dynamic styling
        - **Chatbot only** is fastest for quick Q&A
        
        #### 🎨 SRT Subtitle Features:
        - **Dynamic Font Size**: 20-80 pixels from slider
        - **Custom Colors**: Yellow, White, Red, Green, Blue, Orange, Pink
        - **Font Selection**: Uses TTF fonts from fonts folder
        - **Position Control**: Bottom, Top, Middle placement
        - **Multi-language**: Supports Turkish, English transcription
        
        #### 🔧 Requirements:
        - **Ollama** with `phi4:latest` and `gemma3:4b` models (for AI processing)
        - **Whisper** for SRT subtitle generation
        - **FFmpeg** for audio/video processing
        - **Internet connection** for URL-based videos
        """, elem_classes="main-container")
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True, server_port=7860)