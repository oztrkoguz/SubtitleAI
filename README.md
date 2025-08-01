# 🎥 SubtitleAI 

**AI-powered video subtitle generation and intelligent chat system**

SubtitleAI is an advanced tool that processes YouTube and TikTok videos to generate scene descriptions, translate them, create subtitled videos with text-to-speech narration, and provides an intelligent chatbot for video content analysis. This project leverages state-of-the-art models for video processing, translation, text-to-speech synthesis, and RAG-based question answering.

<img width="1920" height="2208" alt="gradioo" src="https://github.com/user-attachments/assets/e487576b-7bbf-4a9c-a5d4-fffeb567cfcb" />




**Note: The project is currently under active development and will be further enhanced with new features over time.**


**Process Video (AI Descriptions + TTS)**

https://github.com/user-attachments/assets/6cd0071c-0ffb-4779-9677-04b1101539f2

**Generate SRT Subtitles(Whisper Transcription)**



https://github.com/user-attachments/assets/358e9df7-b1c8-4c32-a74a-2aefd82bc126






**Note: The project is currently under active development and will be further enhanced with new features over time.**

## ✨ Features

### 🎬 Video Processing
- **Download YouTube & TikTok Videos**: Automatically download videos from YouTube or TikTok using URLs
- **Scene Detection**: Intelligent detection of scene transitions in videos
- **Frame Description**: Generate English scene descriptions using Gemma3:4b model
- **Multi-language Translation**: Translate descriptions to Turkish using Gemma3:4b model
- **Custom Subtitles**: Create videos with customizable subtitles (font, color, position)
- **Text-to-Speech**: Generate narrated videos using TTS models for English and Turkish
- **Summary Generation**: Provide comprehensive video summaries in selected language

### 📝 SRT Subtitle Generation (NEW!)
- **Whisper Transcription**: Accurate speech-to-text conversion using OpenAI Whisper
- **Custom Styling**: Dynamic font size, colors, and position control
- **Multi-language Support**: Generate subtitles in Turkish, English, and other languages
- **SRT File Export**: Standard SRT format compatible with all video players

### ✨ Advanced Subtitle Effects (NEW!)
- **20+ Animation Effects**: Smooth Fade, Slide (4 directions), Zoom, Pulse, Wave, Shake, Rotate, Bounce, Spiral, Elastic, and more
- **Professional Styling**: Outline thickness (0-8px), shadow effects (0-8px), opacity control (0.1-1.0)
- **Smart Positioning**: 7 position options (corners, edges, center)
- **Mixed Mode**: Random effect combinations for dynamic presentations
- **Font Customization**: TTF font support from fonts folder

### 🤖 Video Chatbot (NEW!)
- **RAG-based Q&A**: Ask questions about video content using advanced RAG system
- **Audio-to-Text**: Convert video audio to text using Whisper or Google Speech Recognition
- **Multi-language Support**: Process videos in English and Turkish
- **Intelligent Responses**: Get contextual answers based on video content
- **Quick Setup**: Setup chatbot without full video processing for faster interaction
- **Similarity Search**: Find relevant content sections in video transcripts

### 🎨 User Interface
- **Modern Gradio Interface**: Clean, responsive web interface
- **Dual Action Mode**: Choose between full video processing or quick chatbot setup
- **Real-time Status**: Live updates on processing status
- **Dynamic Components**: Interface adapts based on selected actions

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/oztrkoguz/SubtitleAI.git
   cd SubtitleAI
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama and required models:**
   ```bash
   # Install Ollama (visit https://ollama.ai for installation)
   ollama pull phi4:latest
   ollama pull gemma3:4b
   ```

5. **Install FFmpeg:**
   - **Windows**: Download from https://ffmpeg.org/download.html
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu**: `sudo apt install ffmpeg`

## 📖 Usage

### 🎬 Video Processing Mode

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Input Video:**
   - Upload a video file, OR
   - Enter YouTube URL, OR
   - Enter TikTok URL

3. **Configure Settings:**
   - Select language (English/Turkish)
   - Customize subtitle settings (font, color, position)
   - **✨ Enable Advanced Effects**: Animations, outline, shadow, opacity

4. **Process Video:**
   - Click "🎬 Process Video" button
   - Wait for AI processing
   - Get subtitled video + summary + chatbot

### 🤖 Quick Chatbot Mode

1. **Input Video URL:**
   - Enter YouTube or TikTok URL

2. **Setup Chatbot:**
   - Select language for audio processing
   - Click "🤖 Setup Chatbot Only" button
   - Wait for audio-to-text conversion

3. **Ask Questions:**
   - Chat interface becomes active
   - Ask questions about video content
   - Get intelligent AI responses

### 📝 SRT Subtitle Generation Mode

1. **Input Video URL:**
   - Enter YouTube or TikTok URL

2. **Configure Subtitle Settings:**
   - Select transcription language (Turkish/English)
   - Choose font size, color, and position
   - **✨ Enable Advanced Effects**: Professional animations and styling

3. **Generate SRT Subtitles:**
   - Click "📝 Generate SRT Subtitles" button
   - Wait for Whisper transcription
   - Get subtitled video + SRT file

### 💬 Example Questions for Chatbot
- "What is the main topic of this video?"
- "Can you summarize the key points?"

## 🏗️ Components

- **app.py**: Main application with Gradio interface and integrated chatbot
- **describe.py**: Video downloading, scene detection, frame description, and translation
- **subtitle.py**: Subtitle creation and video rendering
- **tts.py**: Text-to-speech audio generation
- **video_chat.py**: RAG-based chatbot system with audio-to-text conversion
- **srt_subtitle.py**: SRT subtitle generation with Whisper transcription and advanced effects


## 🤖 Models & Technologies Used

### Video Processing
- **[Gemma3:4b](https://ollama.com/library/gemma3:4b)**: Scene description generation
- **[Gemma3:4b](https://ollama.com/library/gemma3:4b)**: English to Turkish translation
- **[Phi4:latest](https://ollama.com/library/phi4)**: Video content summarization

### Chatbot System
- **[OpenAI Whisper](https://openai.com/research/whisper)**: Audio-to-text conversion
- **[Google Speech Recognition](https://cloud.google.com/speech-to-text)**: Fallback audio processing
- **[FAISS](https://faiss.ai/)**: Vector similarity search
- **[LangChain](https://langchain.com/)**: RAG pipeline and document processing
- **[Sentence Transformers](https://www.sbert.net/)**: Multilingual text embeddings

### Audio & Video
- **[TTS Models](https://huggingface.co/soohyunn/glow-tts)**: Text-to-speech generation
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)**: YouTube/TikTok video downloading
- **[FFmpeg](https://ffmpeg.org/)**: Audio/video processing

## 🔧 Requirements

### System Requirements
- Python 3.8+
- FFmpeg installed and accessible
- Internet connection for URL-based videos
- At least 4GB RAM for optimal performance

### AI Models
- Ollama with phi4:latest and gemma3:4b models
- Whisper model (automatically downloaded)
- Sentence transformers model (automatically downloaded)

## ⚠️ Important Notes

- **Chatbot requires URLs**: Video chatbot only works with YouTube/TikTok URLs, not uploaded files
- **TikTok optimization**: TikTok videos are optimized for short content analysis
- **Language consistency**: Select the same language for both video processing and chatbot for best results
- **Processing time**: Full video processing takes longer than chatbot-only setup

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


