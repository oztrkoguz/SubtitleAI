import yt_dlp
import speech_recognition as sr
import moviepy.editor as mp
import os
import tempfile
from pydub import AudioSegment
import whisper
import subprocess
import sys
from typing import List, Optional, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

# Required imports for RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

class YouTubeToText:
    def __init__(self, enable_rag=False, embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.recognizer = sr.Recognizer()
        # Load Whisper model (for better results)
        self.whisper_model = whisper.load_model("base")
        
        # Setup FFmpeg path
        self.setup_ffmpeg()
        
        # RAG system
        self.enable_rag = enable_rag
        self.embedding_model_name = embedding_model
        self.vector_store = None
        self.qa_chain = None
        self.last_transcript = None
        
        if self.enable_rag:
            self._setup_rag_system()
    
    def setup_ffmpeg(self):
        """Configure FFmpeg"""
        ffmpeg_path = self.find_ffmpeg_path()
        if ffmpeg_path:
            # Add to PATH
            os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
            print(f"✅ FFmpeg found: {ffmpeg_path}")
        else:
            print("❌ FFmpeg not found!")
            print("Solution suggestions:")
            print("1. Download FFmpeg: https://ffmpeg.org/download.html")
            print("2. Extract to C:\\ffmpeg\\bin folder")
            print("3. Or install with conda: conda install ffmpeg")
    
    def find_ffmpeg_path(self):
        """Automatically find FFmpeg path"""
        import shutil
        
        # Search in PATH
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return os.path.dirname(ffmpeg_path)
        
        # Search in common Windows paths
        common_paths = [
            "C:\\ffmpeg\\bin",
            "C:\\Program Files\\ffmpeg\\bin",
            "C:\\Program Files (x86)\\ffmpeg\\bin",
            os.path.join(os.getcwd(), "ffmpeg", "bin")
        ]
        
        for path in common_paths:
            if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                return path
        
        return None

    def test_ffmpeg(self):
        """Test if FFmpeg is working"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ FFmpeg test successful")
                return True
        except Exception as e:
            print(f"❌ FFmpeg test failed: {e}")
        return False

    def download_audio(self, youtube_url, output_path="temp_audio.wav"):
        """Download audio file from YouTube video"""
        try:
            # Test FFmpeg
            if not self.test_ffmpeg():
                print("FFmpeg not working, trying alternative format...")
                return self.download_audio_alternative(youtube_url, output_path)
            
            # Automatically find FFmpeg path or specify manually
            ffmpeg_path = self.find_ffmpeg_path()
            if not ffmpeg_path:
                print("FFmpeg not found! Specifying manual path...")
                ffmpeg_path = "C:\\ffmpeg\\bin"  # Manual path
            
            # Get full file path
            abs_output_path = os.path.abspath(output_path)
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': abs_output_path.replace('.wav', '.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'ffmpeg_location': ffmpeg_path,  # Specify FFmpeg path
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            
            # Check if file was actually created
            if os.path.exists(abs_output_path):
                print(f"✅ Audio file created: {abs_output_path}")
                return abs_output_path
            else:
                print(f"❌ Audio file could not be created: {abs_output_path}")
                # Check alternative file names
                for possible_ext in ['.wav', '.webm', '.m4a', '.mp3']:
                    possible_file = abs_output_path.replace('.wav', possible_ext)
                    if os.path.exists(possible_file):
                        print(f"✅ Alternative file found: {possible_file}")
                        return possible_file
                return None
        except Exception as e:
            print(f"Audio download error: {e}")
            return self.download_audio_alternative(youtube_url, output_path)

    def download_audio_alternative(self, youtube_url, output_path="temp_audio"):
        """Alternative audio download without FFmpeg"""
        try:
            print("🔄 Trying alternative download method...")
            
            abs_output_path = os.path.abspath(output_path)
            
            # Download only audio file, no conversion
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio',
                'outtmpl': abs_output_path + '.%(ext)s',
                'nopostprocessors': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                ydl.download([youtube_url])
                
                # Find the extension of downloaded file
                ext = ydl.prepare_filename(info).split('.')[-1]
                downloaded_file = f"{abs_output_path}.{ext}"
                
                if os.path.exists(downloaded_file):
                    print(f"✅ Raw audio file downloaded: {downloaded_file}")
                    return downloaded_file
                    
            return None
        except Exception as e:
            print(f"Alternative download error: {e}")
            return None
    
    def audio_to_text_whisper(self, audio_path, language='en'):
        """Convert audio file to text using Whisper (recommended)"""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"❌ Audio file not found: {audio_path}")
                return None
            
            print(f"🎵 Processing with Whisper: {audio_path}")
            print(f"📊 File size: {os.path.getsize(audio_path) / 1024 / 1024:.2f} MB")
            print(f"🌍 Language setting: {language}")
            
            # Add FFmpeg path to environment for Whisper
            ffmpeg_path = self.find_ffmpeg_path()
            if ffmpeg_path and ffmpeg_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
            
            # Set Whisper language codes
            whisper_lang = language
            if language == 'tr':
                whisper_lang = 'tr'
            elif language == 'en':
                whisper_lang = 'en'
            
            # Add verbose parameter to Whisper to see errors better
            result = self.whisper_model.transcribe(
                audio_path, 
                language=whisper_lang,  # Dynamic language support
                verbose=True,
                fp16=False  # For compatibility
            )
            
            return result["text"]
        except Exception as e:
            print(f"Whisper conversion error: {e}")
            print("🔄 Trying Google Speech Recognition...")
            return self.audio_to_text_google_fallback(audio_path, language)
    
    def audio_to_text_google_fallback(self, audio_path, language='en'):
        """Use Google Speech Recognition if Whisper fails"""
        try:
            # Convert audio file to wav format using Pydub
            if not audio_path.endswith('.wav'):
                print("🔄 Converting audio file to WAV format...")
                audio = AudioSegment.from_file(audio_path)
                wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
                audio.export(wav_path, format="wav")
                audio_path = wav_path
                print(f"✅ WAV file created: {wav_path}")
            
            return self.audio_to_text_google(audio_path, language)
        except Exception as e:
            print(f"Fallback conversion error: {e}")
            return None
    
    def audio_to_text_google(self, audio_path, language='en'):
        """Convert audio file to text using Google Speech Recognition"""
        try:
            # Load audio file
            audio = AudioSegment.from_wav(audio_path)
            
            # Split large files into chunks (60 seconds)
            chunk_length_ms = 60000
            chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
            # Google Speech Recognition language codes
            google_lang = "en-US"
            if language == 'tr':
                google_lang = "tr-TR"
            elif language == 'en':
                google_lang = "en-US"
            
            print(f"🌍 Google Speech language setting: {google_lang}")
            
            full_text = ""
            
            for i, chunk in enumerate(chunks):
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    chunk.export(temp_file.name, format="wav")
                    
                    # Load audio file
                    with sr.AudioFile(temp_file.name) as source:
                        audio_data = self.recognizer.record(source)
                    
                    try:
                        # Convert to text with dynamic language support
                        text = self.recognizer.recognize_google(audio_data, language=google_lang)
                        full_text += text + " "
                        print(f"Chunk {i+1} processed: {text[:50]}...")
                    except sr.UnknownValueError:
                        print(f"Chunk {i+1} could not be understood")
                    except sr.RequestError as e:
                        print(f"Google API error: {e}")
                    
                    # Delete temporary file
                    os.unlink(temp_file.name)
            
            return full_text.strip()
            
        except Exception as e:
            print(f"Google Speech Recognition error: {e}")
            return None
    
    def process_youtube_video(self, youtube_url, method="whisper", language='en'):
        """Process YouTube video and convert to text"""
        print(f"Processing: {youtube_url}")
        print(f"🌍 Language: {language}")
        
        # Download audio file
        audio_path = self.download_audio(youtube_url)
        if not audio_path:
            return None
        
        # Convert to text
        if method == "whisper":
            text = self.audio_to_text_whisper(audio_path, language)
        else:
            text = self.audio_to_text_google_fallback(audio_path, language)
        
        # Clean up temporary files
        self.cleanup_temp_files(audio_path)
        
        return text
    
    def cleanup_temp_files(self, audio_path):
        """Clean up temporary files"""
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"🗑️ Temporary file deleted: {audio_path}")
            
            # Clean up other possible extensions
            base_path = audio_path.rsplit('.', 1)[0] if audio_path else "temp_audio"
            for ext in ['.webm', '.m4a', '.mp3', '.wav']:
                temp_file = base_path + ext
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"🗑️ File deleted: {temp_file}")
        except Exception as e:
            print(f"⚠️ File deletion error: {e}")
    
    def save_text_to_file(self, text, filename="transcript.txt"):
        """Save text to file"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"💾 Text saved: {filename}")
        except Exception as e:
            print(f"File saving error: {e}")
    
    # =============================================================================
    # RAG SYSTEM FUNCTIONS
    # =============================================================================
    
    def _setup_rag_system(self):
        """Initialize RAG system"""
        
        try:
            # Multilingual embedding model
            print(f"🔧 Initializing RAG system...")
            print(f"📚 Embedding model: {self.embedding_model_name}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # LLM (Ollama)
            try:
                self.llm = ChatOllama(
                    model="phi4:latest",  # Default model
                    temperature=0.1,
                    num_ctx=4096,
                    num_predict=512,
                    verbose=False
                )
                print("✅ RAG system ready!")
            except Exception as e:
                print(f"⚠️ LLM could not be loaded: {e}")
                print("💡 Ollama not installed - only similarity search available")
                self.llm = None
                
        except Exception as e:
            print(f"❌ RAG system error: {e}")
            self.enable_rag = False
    
    def create_rag_from_transcript(self, transcript: str, metadata: Dict = None) -> bool:
        """Create RAG system from transcript"""
        if not self.enable_rag:
            print("❌ RAG system not active")
            return False
        
        try:
            print("🔧 Creating RAG vector store...")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(transcript)
            print(f"📝 Text split into {len(chunks)} chunks")
            
            # Create document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "source": "youtube_transcript"
                }
                if metadata:
                    doc_metadata.update(metadata)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.last_transcript = transcript
            
            print("✅ RAG vector store created!")
            return True
            
        except Exception as e:
            print(f"❌ RAG creation error: {e}")
            return False
    
    def setup_qa_system(self):
        """Setup question-answer system"""
        if not self.enable_rag or not self.vector_store or not self.llm:
            print("❌ QA system cannot be setup - requirements not met")
            return False
        
        try:
            # English prompt template
            prompt_template = """
Use the following context information to answer the question. Answer strictly based on the provided context.

Do not include phrases like "in this context" or "according to the context" in your response. Simply provide a direct, natural answer as if you have comprehensive knowledge about the topic.

If the answer is not found in the provided context, say "This information is not found in the text".

Context:
{context}

Question: {question}

Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # RetrievalQA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            print("✅ Question-answer system ready!")
            return True
            
        except Exception as e:
            print(f"❌ QA system error: {e}")
            return False
    
    def ask_question(self, question: str) -> Dict:
        """Ask question about transcript"""
        if not self.enable_rag or not self.vector_store:
            return {
                "error": "RAG system not active or vector store missing",
                "question": question,
                "answer": None
            }
        
        # Setup QA system (if not already setup)
        if not self.qa_chain:
            if not self.setup_qa_system():
                return {
                    "error": "QA system could not be setup",
                    "question": question,
                    "answer": None
                }
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "error": None
            }
            
            return response
            
        except Exception as e:
            return {
                "error": str(e),
                "question": question,
                "answer": None
            }
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Perform similarity search"""
        if not self.enable_rag or not self.vector_store:
            print("❌ RAG system not active")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            
            print(f"🔍 {len(results)} results found for '{query}':")
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.page_content[:100]}...")
            
            return results
            
        except Exception as e:
            print(f"❌ Similarity search error: {e}")
            return []
    
    def save_rag_system(self, path: str = "./vector_store"):
        """Save RAG system"""
        if not self.enable_rag or not self.vector_store:
            print("❌ No RAG system to save")
            return False
        
        try:
            os.makedirs(path, exist_ok=True)
            self.vector_store.save_local(path)
            
            # Also save transcript
            with open(os.path.join(path, "transcript.txt"), "w", encoding="utf-8") as f:
                f.write(self.last_transcript or "")
            
            print(f"💾 RAG system saved: {path}")
            return True
            
        except Exception as e:
            print(f"❌ RAG saving error: {e}")
            return False
    
    def load_rag_system(self, path: str = "./vector_store") -> bool:
        """Load RAG system"""
        if not self.enable_rag:
            print("❌ RAG system not active")
            return False
        
        if not os.path.exists(path):
            print(f"❌ RAG system not found: {path}")
            return False
        
        try:
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Also load transcript
            transcript_path = os.path.join(path, "transcript.txt")
            if os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8") as f:
                    self.last_transcript = f.read()
            
            print(f"✅ RAG system loaded: {path}")
            return True
            
        except Exception as e:
            print(f"❌ RAG loading error: {e}")
            return False
    
    def process_with_rag(self, youtube_url: str, method: str = "whisper", language: str = 'en') -> Dict:
        """Process YouTube video and analyze with RAG system"""
        
        # 1. Get normal transcript
        print(f"🎬 Processing YouTube video: {youtube_url}")
        transcript = self.process_youtube_video(youtube_url, method=method, language=language)
        
        if not transcript:
            return {"error": "Transcript could not be obtained", "transcript": None, "rag_ready": False}
        
        result = {
            "transcript": transcript,
            "rag_ready": False,
            "error": None
        }
        
        # 2. Create RAG system
        if self.enable_rag:
            metadata = {
                "source": "youtube",
                "url": youtube_url,
                "method": method,
                "language": language
            }
            
            if self.create_rag_from_transcript(transcript, metadata):
                result["rag_ready"] = True
        
        return result

def main():
    """Main RAG system"""
    print("🚀 YouTube RAG System")
    
    # YouTube URL (enter your desired URL here)
    video_url = "https://www.youtube.com/watch?v=Ht2QW5PV-eY"
    print(f"📺 Video to process: {video_url}")
    
    # Start with RAG system
    converter = YouTubeToText(
        enable_rag=True,
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Process video and create RAG system
    print("\n" + "="*60)
    result = converter.process_with_rag(video_url, method="whisper")
    
    if not result["transcript"]:
        print("❌ Transcript could not be obtained!")
        return
    
    if not result["rag_ready"]:
        print("❌ RAG system could not be created!")
        return
    
    print("\n" + "="*60)
    print("📝 TRANSCRIPT:")
    print("="*60)
    print(result["transcript"])
    
    # Save RAG system
    converter.save_rag_system("./rag_data")
    
    # Question-answer loop
    print("\n" + "="*60)
    print("💬 QUESTION-ANSWER SYSTEM (quit to exit)")
    print("="*60)
    print("Ask your questions about the video...")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Question-answer
            answer = converter.ask_question(question)
            if not answer.get("error"):
                print(f"\n💬 Answer: {answer['answer']}")
            else:
                print(f"\n❌ Error: {answer['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()