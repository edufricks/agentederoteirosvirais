import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import openai
import tempfile
import re
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -------- FUNÇÕES AUXILIARES --------

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_transcript_youtube(video_id, lang="pt"):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        return transcript
    except:
        return None

def download_audio(video_url):
    """Baixa o áudio do YouTube usando yt_dlp"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_file.name,
        'quiet': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return temp_file.name

def transcribe_whisper_api(audio_path):
    """Transcreve áudio com Whisper API (OpenAI Cloud)"""
    with open(audio_path, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )
    return transcript["segments"]

def transcript_to_text(transcript):
    """Transforma transcrição em texto legível"""
    return " ".join([f"{t['start']:.2f}s: {t['text']}" for t in transcript])

def generate_script_with_gpt(transcription):
    """Reescreve a transcrição no formato viral"""
    prompt = f"""
Você é um especialista em roteiros virais de YouTube.  
Reescreva o vídeo abaixo no formato do seguinte diagrama:

1. **5 segundos iniciais (gancho explosivo refletindo a thumb)**  
   - Sugira também recursos visuais/mosaicos que reforcem.  

2. **30 segundos de contexto e questionamento**  

3. **90 segundos alternando entre momentos opostos**  

4. **Resposta superando expectativas**  

5. **Opinião final**  

6. **Fechamento (CTA)**  

Além disso, sugira:  
- **Título viral** (máx. 60 caracteres)  
- **Ideia de Thumbnail**  
- **Clipes curtos (Shorts/TikTok)** com minutagem  
- **Sugestões de edição** (inserts, prints, mosaicos etc).  

Transcrição:  
{transcription}
"""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message["content"]

def export_docx(text):
    doc = Document()
    doc.add_paragraph(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(temp_file.name)
    return temp_file.name

def export_pdf(text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name)
    styles = getSampleStyleSheet()
    story = [Paragraph(p, styles["Normal"]) for p in text.split("\n")]
    for s in story:
        s.spaceAfter = 12
    doc.build(story)
    return temp_file.name

def export_srt(transcript):
    srt_content = ""
    for i, t in enumerate(transcript, start=1):
        start = t['start']
        end = t['start'] + t['duration']
        srt_content += f"{i}\n"
        srt_content += f"{format_srt_time(start)} --> {format_srt_time(end)}\n"
        srt_content += t['text'] + "\n\n"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".srt")
    with open(temp_file.name, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return temp_file.name

def format_srt_time(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

# -------- INTERFACE STREAMLIT --------

st.set_page_config(page_title="Agente de Roteiros Virais", layout="wide")
st.title("🎬 Agente de Roteiros Virais")

# Campo para chave da OpenAI
api_key = st.text_input("🔑 Insira sua chave da OpenAI:", type="password")

if not api_key:
    st.warning("⚠️ Insira sua chave da OpenAI para continuar.")
    st.stop()

openai.api_key = api_key

url = st.text_input("Cole o link do vídeo do YouTube:")

if url:
    video_id = extract_video_id(url)
    if video_id:
        st.video(f"https://www.youtube.com/watch?v={video_id}")

        with st.spinner("Buscando transcrição..."):
            transcript = get_transcript_youtube(video_id)

        if not transcript:
            st.warning("❌ Não encontrei legendas. Vou transcrever com Whisper API...")
            with st.spinner("Baixando áudio e transcrevendo..."):
                audio_path = download_audio(url)
                transcript = transcribe_whisper_api(audio_path)

        # Exibir transcrição
        st.subheader("📝 Transcrição")
        text_transcript = transcript_to_text(transcript)
        with st.expander("Ver transcrição completa"):
            st.write(text_transcript)

        if st.button("Gerar Roteiro no Formato Viral"):
            with st.spinner("Gerando roteiro com GPT..."):
                roteiro = generate_script_with_gpt(text_transcript)

            st.subheader("📜 Roteiro Final")
            st.markdown(roteiro)

            # Exportações
            docx_file = export_docx(roteiro)
            pdf_file = export_pdf(roteiro)
            srt_file = export_srt(transcript)

            st.download_button("⬇️ Baixar DOCX", data=open(docx_file, "rb"), file_name="roteiro.docx")
            st.download_button("⬇️ Baixar PDF", data=open(pdf_file, "rb"), file_name="roteiro.pdf")
            st.download_button("⬇️ Baixar SRT", data=open(srt_file, "rb"), file_name="transcricao.srt")
