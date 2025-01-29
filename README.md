# 🎥 SubtitleAI
SubtitleAI is a tool that processes YouTube videos to generate scene descriptions, translate them, and create a subtitled video with text-to-speech narration. This project leverages state-of-the-art models for video processing, translation, and text-to-speech synthesis.

![image](https://github.com/user-attachments/assets/b363a364-edcf-4c42-90f7-c0d941e0aef3)

## Features

- **Download YouTube Videos**: Automatically download videos from YouTube using a URL.
- **Scene Detection**: Detects scene transitions in the video.
- **Frame Description**: Generates English descriptions for each scene using a pre-trained model.
- **Translation**: Translates descriptions to Turkish using the M2M100 model.
- **Subtitled Video**: Creates a video with subtitles based on the translated descriptions.
- **Text-to-Speech**: Generates a narrated video using TTS models for English and Turkish.
- **Summary Generation**: Provides a summary of the video content in the selected language using Ollama.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/youtube-video-summarizer.git
   cd youtube-video-summarizer
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

## Usage

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Upload a video or enter a YouTube URL**: You can either upload a video file or provide a YouTube URL for processing.

3. **Select Language**: Choose the language for descriptions and voice (English or Turkish).

4. **Customize Subtitles**: Adjust font size, color, and position for subtitles.

5. **Process Video**: Click the "Process Video" button to start the processing.

6. **Output**: The processed video with subtitles and narration will be generated, along with a text summary.

## Components

- **describe.py**: Handles video downloading, scene detection, frame description, and translation.
- **subtitle.py**: Manages subtitle creation and video rendering.
- **app.py**: Main application logic using Gradio for the user interface.
- **tts.py**: Generates text-to-speech audio for the video.
- **requirements.txt**: Lists all the dependencies required for the project.

## Models Used

- **MiniCPM**: For generating scene descriptions.[MiniCPM](https://huggingface.co/openbmb/MiniCPM-o-2_6)
- **M2M100**: For translating descriptions from English to Turkish.[M2M100](https://huggingface.co/facebook/m2m100_418M)
- **TTS Models**: For generating audio narration in English and Turkish.[TTS Models](https://huggingface.co/soohyunn/glow-tts)
- **Ollama**: For generating a coherent summary of the video content.[Mistral](https://ollama.com/library/mistral)


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or support,open an issue on GitHub.
