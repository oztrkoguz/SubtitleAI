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

# RAG için gerekli importlar
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
        # Whisper modelini yükle (daha iyi sonuçlar için)
        self.whisper_model = whisper.load_model("base")
        
        # FFmpeg yolunu ayarla
        self.setup_ffmpeg()
        
        # RAG sistemi
        self.enable_rag = enable_rag
        self.embedding_model_name = embedding_model
        self.vector_store = None
        self.qa_chain = None
        self.last_transcript = None
        
        if self.enable_rag:
            self._setup_rag_system()
    
    def setup_ffmpeg(self):
        """FFmpeg'i yapılandır"""
        ffmpeg_path = self.find_ffmpeg_path()
        if ffmpeg_path:
            # PATH'e ekle
            os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
            print(f"✅ FFmpeg bulundu: {ffmpeg_path}")
        else:
            print("❌ FFmpeg bulunamadı!")
            print("Çözüm önerileri:")
            print("1. FFmpeg'i indirin: https://ffmpeg.org/download.html")
            print("2. C:\\ffmpeg\\bin klasörüne çıkarın")
            print("3. Veya conda ile kurun: conda install ffmpeg")
    
    def find_ffmpeg_path(self):
        """FFmpeg yolunu otomatik bul"""
        import shutil
        
        # PATH'te ara
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return os.path.dirname(ffmpeg_path)
        
        # Yaygın Windows yollarında ara
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
        """FFmpeg'in çalışıp çalışmadığını test et"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ FFmpeg test başarılı")
                return True
        except Exception as e:
            print(f"❌ FFmpeg test başarısız: {e}")
        return False

    def download_audio(self, youtube_url, output_path="temp_audio.wav"):
        """YouTube videosundan ses dosyasını indir"""
        try:
            # FFmpeg test et
            if not self.test_ffmpeg():
                print("FFmpeg çalışmıyor, alternatif format denenecek...")
                return self.download_audio_alternative(youtube_url, output_path)
            
            # FFmpeg yolunu otomatik bul veya manuel belirt
            ffmpeg_path = self.find_ffmpeg_path()
            if not ffmpeg_path:
                print("FFmpeg bulunamadı! Manuel yol belirtiliyor...")
                ffmpeg_path = "C:\\ffmpeg\\bin"  # Manuel yol
            
            # Tam dosya yolunu al
            abs_output_path = os.path.abspath(output_path)
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': abs_output_path.replace('.wav', '.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'ffmpeg_location': ffmpeg_path,  # FFmpeg yolunu belirt
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            
            # Dosyanın gerçekten oluştuğunu kontrol et
            if os.path.exists(abs_output_path):
                print(f"✅ Ses dosyası oluşturuldu: {abs_output_path}")
                return abs_output_path
            else:
                print(f"❌ Ses dosyası oluşturulamadı: {abs_output_path}")
                # Alternatif dosya adlarını kontrol et
                for possible_ext in ['.wav', '.webm', '.m4a', '.mp3']:
                    possible_file = abs_output_path.replace('.wav', possible_ext)
                    if os.path.exists(possible_file):
                        print(f"✅ Alternatif dosya bulundu: {possible_file}")
                        return possible_file
                return None
        except Exception as e:
            print(f"Ses indirme hatası: {e}")
            return self.download_audio_alternative(youtube_url, output_path)

    def download_audio_alternative(self, youtube_url, output_path="temp_audio"):
        """FFmpeg olmadan ses indirme alternatifi"""
        try:
            print("🔄 Alternatif indirme yöntemi deneniyor...")
            
            abs_output_path = os.path.abspath(output_path)
            
            # Sadece ses dosyasını indir, dönüştürme yapma
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio',
                'outtmpl': abs_output_path + '.%(ext)s',
                'nopostprocessors': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                ydl.download([youtube_url])
                
                # İndirilen dosyanın uzantısını bul
                ext = ydl.prepare_filename(info).split('.')[-1]
                downloaded_file = f"{abs_output_path}.{ext}"
                
                if os.path.exists(downloaded_file):
                    print(f"✅ Ham ses dosyası indirildi: {downloaded_file}")
                    return downloaded_file
                    
            return None
        except Exception as e:
            print(f"Alternatif indirme hatası: {e}")
            return None
    
    def audio_to_text_whisper(self, audio_path, language='en'):
        """Whisper kullanarak ses dosyasını metne dönüştür (önerilen)"""
        try:
            # Dosyanın varlığını kontrol et
            if not os.path.exists(audio_path):
                print(f"❌ Ses dosyası bulunamadı: {audio_path}")
                return None
            
            print(f"🎵 Whisper ile işleniyor: {audio_path}")
            print(f"📊 Dosya boyutu: {os.path.getsize(audio_path) / 1024 / 1024:.2f} MB")
            print(f"🌍 Dil ayarı: {language}")
            
            # Whisper için FFmpeg yolunu environment'a ekle
            ffmpeg_path = self.find_ffmpeg_path()
            if ffmpeg_path and ffmpeg_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
            
            # Whisper dil kodlarını ayarla
            whisper_lang = language
            if language == 'tr':
                whisper_lang = 'tr'
            elif language == 'en':
                whisper_lang = 'en'
            
            # Whisper'a verbose parametre ekleyerek hatayı daha iyi görelim
            result = self.whisper_model.transcribe(
                audio_path, 
                language=whisper_lang,  # Dinamik dil desteği
                verbose=True,
                fp16=False  # Uyumluluk için
            )
            
            return result["text"]
        except Exception as e:
            print(f"Whisper dönüşüm hatası: {e}")
            print("🔄 Google Speech Recognition deneniyor...")
            return self.audio_to_text_google_fallback(audio_path, language)
    
    def audio_to_text_google_fallback(self, audio_path, language='en'):
        """Whisper başarısız olursa Google Speech Recognition kullan"""
        try:
            # Pydub ile ses dosyasını wav formatına dönüştür
            if not audio_path.endswith('.wav'):
                print("🔄 Ses dosyası WAV formatına dönüştürülüyor...")
                audio = AudioSegment.from_file(audio_path)
                wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
                audio.export(wav_path, format="wav")
                audio_path = wav_path
                print(f"✅ WAV dosyası oluşturuldu: {wav_path}")
            
            return self.audio_to_text_google(audio_path, language)
        except Exception as e:
            print(f"Fallback dönüşüm hatası: {e}")
            return None
    
    def audio_to_text_google(self, audio_path, language='en'):
        """Google Speech Recognition ile ses dosyasını metne dönüştür"""
        try:
            # Ses dosyasını yükle
            audio = AudioSegment.from_wav(audio_path)
            
            # Büyük dosyaları parçalara böl (60 saniye)
            chunk_length_ms = 60000
            chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
            # Google Speech Recognition dil kodları
            google_lang = "en-US"
            if language == 'tr':
                google_lang = "tr-TR"
            elif language == 'en':
                google_lang = "en-US"
            
            print(f"🌍 Google Speech dil ayarı: {google_lang}")
            
            full_text = ""
            
            for i, chunk in enumerate(chunks):
                # Geçici dosya oluştur
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    chunk.export(temp_file.name, format="wav")
                    
                    # Ses dosyasını yükle
                    with sr.AudioFile(temp_file.name) as source:
                        audio_data = self.recognizer.record(source)
                    
                    try:
                        # Dinamik dil desteği ile metne dönüştür
                        text = self.recognizer.recognize_google(audio_data, language=google_lang)
                        full_text += text + " "
                        print(f"Parça {i+1} işlendi: {text[:50]}...")
                    except sr.UnknownValueError:
                        print(f"Parça {i+1} anlaşılamadı")
                    except sr.RequestError as e:
                        print(f"Google API hatası: {e}")
                    
                    # Geçici dosyayı sil
                    os.unlink(temp_file.name)
            
            return full_text.strip()
            
        except Exception as e:
            print(f"Google Speech Recognition hatası: {e}")
            return None
    
    def process_youtube_video(self, youtube_url, method="whisper", language='en'):
        """YouTube videosunu işleyip metne dönüştür"""
        print(f"İşleniyor: {youtube_url}")
        print(f"🌍 Dil: {language}")
        
        # Ses dosyasını indir
        audio_path = self.download_audio(youtube_url)
        if not audio_path:
            return None
        
        # Metne dönüştür
        if method == "whisper":
            text = self.audio_to_text_whisper(audio_path, language)
        else:
            text = self.audio_to_text_google_fallback(audio_path, language)
        
        # Geçici dosyaları temizle
        self.cleanup_temp_files(audio_path)
        
        return text
    
    def cleanup_temp_files(self, audio_path):
        """Geçici dosyaları temizle"""
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"🗑️ Geçici dosya silindi: {audio_path}")
            
            # Diğer olası uzantıları da temizle
            base_path = audio_path.rsplit('.', 1)[0] if audio_path else "temp_audio"
            for ext in ['.webm', '.m4a', '.mp3', '.wav']:
                temp_file = base_path + ext
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"🗑️ Dosya silindi: {temp_file}")
        except Exception as e:
            print(f"⚠️ Dosya silme hatası: {e}")
    
    def save_text_to_file(self, text, filename="transcript.txt"):
        """Metni dosyaya kaydet"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"💾 Metin kaydedildi: {filename}")
        except Exception as e:
            print(f"Dosya kaydetme hatası: {e}")
    
    # =============================================================================
    # RAG SİSTEMİ FONKSİYONLARI
    # =============================================================================
    
    def _setup_rag_system(self):
        """RAG sistemini başlat"""
        
        try:
            # Çok dilli embedding modeli
            print(f"🔧 RAG sistemi başlatılıyor...")
            print(f"📚 Embedding modeli: {self.embedding_model_name}")
            
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
                    model="phi4:latest",  # Varsayılan model
                    temperature=0.1,
                    num_ctx=4096,
                    num_predict=512,
                    verbose=False
                )
                print("✅ RAG sistemi hazır!")
            except Exception as e:
                print(f"⚠️ LLM yüklenemedi: {e}")
                print("💡 Ollama kurulu değil - sadece benzerlik araması kullanılabilir")
                self.llm = None
                
        except Exception as e:
            print(f"❌ RAG sistem hatası: {e}")
            self.enable_rag = False
    
    def create_rag_from_transcript(self, transcript: str, metadata: Dict = None) -> bool:
        """Transkriptten RAG sistemi oluştur"""
        if not self.enable_rag:
            print("❌ RAG sistemi aktif değil")
            return False
        
        try:
            print("🔧 RAG vector store oluşturuluyor...")
            
            # Metni parçalara böl
            chunks = self.text_splitter.split_text(transcript)
            print(f"📝 Metin {len(chunks)} parçaya bölündü")
            
            # Document objelerini oluştur
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
            
            # FAISS vector store oluştur
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.last_transcript = transcript
            
            print("✅ RAG vector store oluşturuldu!")
            return True
            
        except Exception as e:
            print(f"❌ RAG oluşturma hatası: {e}")
            return False
    
    def setup_qa_system(self):
        """Soru-cevap sistemini kur"""
        if not self.enable_rag or not self.vector_store or not self.llm:
            print("❌ QA sistemi kurulamıyor - gereksinimler karşılanmıyor")
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
            
            # RetrievalQA zinciri
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
            
            print("✅ Soru-cevap sistemi hazır!")
            return True
            
        except Exception as e:
            print(f"❌ QA sistem hatası: {e}")
            return False
    
    def ask_question(self, question: str) -> Dict:
        """Transkript hakkında soru sor"""
        if not self.enable_rag or not self.vector_store:
            return {
                "error": "RAG sistemi aktif değil veya vector store yok",
                "question": question,
                "answer": None
            }
        
        # QA sistemini kur (eğer kurulu değilse)
        if not self.qa_chain:
            if not self.setup_qa_system():
                return {
                    "error": "QA sistemi kurulamadı",
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
        """Benzerlik araması yap"""
        if not self.enable_rag or not self.vector_store:
            print("❌ RAG sistemi aktif değil")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            
            print(f"🔍 '{query}' için {len(results)} sonuç bulundu:")
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.page_content[:100]}...")
            
            return results
            
        except Exception as e:
            print(f"❌ Benzerlik arama hatası: {e}")
            return []
    
    def save_rag_system(self, path: str = "./vector_store"):
        """RAG sistemini kaydet"""
        if not self.enable_rag or not self.vector_store:
            print("❌ Kaydedilecek RAG sistemi yok")
            return False
        
        try:
            os.makedirs(path, exist_ok=True)
            self.vector_store.save_local(path)
            
            # Transcript'i de kaydet
            with open(os.path.join(path, "transcript.txt"), "w", encoding="utf-8") as f:
                f.write(self.last_transcript or "")
            
            print(f"💾 RAG sistemi kaydedildi: {path}")
            return True
            
        except Exception as e:
            print(f"❌ RAG kaydetme hatası: {e}")
            return False
    
    def load_rag_system(self, path: str = "./vector_store") -> bool:
        """RAG sistemini yükle"""
        if not self.enable_rag:
            print("❌ RAG sistemi aktif değil")
            return False
        
        if not os.path.exists(path):
            print(f"❌ RAG sistemi bulunamadı: {path}")
            return False
        
        try:
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Transcript'i de yükle
            transcript_path = os.path.join(path, "transcript.txt")
            if os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8") as f:
                    self.last_transcript = f.read()
            
            print(f"✅ RAG sistemi yüklendi: {path}")
            return True
            
        except Exception as e:
            print(f"❌ RAG yükleme hatası: {e}")
            return False
    
    def process_with_rag(self, youtube_url: str, method: str = "whisper", language: str = 'en') -> Dict:
        """YouTube videosunu işleyip RAG sistemi ile analiz et"""
        
        # 1. Normal transkript al
        print(f"🎬 YouTube videosu işleniyor: {youtube_url}")
        transcript = self.process_youtube_video(youtube_url, method=method, language=language)
        
        if not transcript:
            return {"error": "Transkript alınamadı", "transcript": None, "rag_ready": False}
        
        result = {
            "transcript": transcript,
            "rag_ready": False,
            "error": None
        }
        
        # 2. RAG sistemi oluştur
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
    """Ana RAG sistemi"""
    print("🚀 YouTube RAG Sistemi")
    
    # YouTube URL'si (buraya istediğiniz URL'yi girin)
    video_url = "https://www.youtube.com/watch?v=Ht2QW5PV-eY"
    print(f"📺 İşlenecek video: {video_url}")
    
    # RAG sistemi ile başlat
    converter = YouTubeToText(
        enable_rag=True,
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Video işle ve RAG sistemi oluştur
    print("\n" + "="*60)
    result = converter.process_with_rag(video_url, method="whisper")
    
    if not result["transcript"]:
        print("❌ Transkript alınamadı!")
        return
    
    if not result["rag_ready"]:
        print("❌ RAG sistemi oluşturulamadı!")
        return
    
    print("\n" + "="*60)
    print("📝 TRANSKRIPT:")
    print("="*60)
    print(result["transcript"])
    
    # RAG sistemini kaydet
    converter.save_rag_system("./rag_data")
    
    # Soru-cevap döngüsü
    print("\n" + "="*60)
    print("💬 SORU-CEVAP SİSTEMİ (quit ile çık)")
    print("="*60)
    print("Video hakkında sorularınızı sorun...")
    
    while True:
        try:
            question = input("\n❓ Sorunuz: ").strip()
            
            if question.lower() in ['quit', 'çık', 'exit', 'q']:
                print("👋 Görüşürüz!")
                break
            
            if not question:
                continue
            
            # Soru-cevap
            answer = converter.ask_question(question)
            if not answer.get("error"):
                print(f"\n💬 Cevap: {answer['answer']}")
            else:
                print(f"\n❌ Hata: {answer['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 Görüşürüz!")
            break

if __name__ == "__main__":
    main()