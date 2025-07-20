# 🎥 SubtitleAI
SubtitleAI is a tool that processes YouTube videos to generate scene descriptions, translate them, and create a subtitled video with text-to-speech narration. This project leverages state-of-the-art models for video processing, translation, and text-to-speech synthesis.



<img width="1920" height="1970" alt="images" src="https://github.com/user-attachments/assets/4f6355b2-9f36-40e9-aed5-d7c76a461e11" />








https://github.com/user-attachments/assets/6cd0071c-0ffb-4779-9677-04b1101539f2






**Note: The project is currently under active development and will be further enhanced with new features over time.**


## Features

- **Download YouTube & TikTok Videos:**: Automatically download videos from YouTube or TikTok using a URL.
- **Scene Detection**: Detects scene transitions in the video.
- **Frame Description**: Generates English scene descriptions using the Gemma3:4b model fine-tuned or prompted for image captioning tasks.
- **Translation**: Translates the English descriptions to Turkish using the Gemma3:4b model, guided for multilingual generation.
- **Subtitled Video**: Creates a video with subtitles based on the translated descriptions.
- **Text-to-Speech**: Generates a narrated video using TTS models for English and Turkish.
- **Summary Generation**: Provides a summary of the video content in the selected language using Ollama.

## Installation

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

## Usage

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Upload a video or enter a YouTube/TikTok URL**: You can either upload a video file or provide a video URL from YouTube or TikTok.

3. **Select Language**: Choose the language for descriptions and voice (English or Turkish).

4. **Customize Subtitles**: Adjust font size, color, and position for subtitles.

5. **Process Video**: Click the "Process Video" button to start the processing.

6. **Output**: The processed video with subtitles and narration will be generated, along with a text summary.

## Components

- **describe.py**: Handles video downloading, scene detection, frame description, and translation.
- **subtitle.py**: Manages subtitle creation and video rendering.
- **app.py**: Main application logic using Gradio for the user interface.
- **tts.py**: Generates text-to-speech audio for the video.

## Models Used

- **Gemma3:4b**: For generating scene descriptions.[Gemma3:4b](https://ollama.com/library/gemma3:4b)
- **Gemma3:4b**: For translating descriptions from English to Turkish.[Gemma3:4b](https://ollama.com/library/gemma3:4b)
- **TTS Models**: For generating audio narration in English and Turkish.[TTS Models](https://huggingface.co/soohyunn/glow-tts)
- **Ollama**: For generating a coherent summary of the video content.[Mistral](https://ollama.com/library/mistral)


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or support,open an issue on GitHub.
