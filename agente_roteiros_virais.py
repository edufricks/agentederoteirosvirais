# app.py
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import openai
import tempfile
import re
import os

from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Configuração da API
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -------- FUNÇÕES --------

def extract_video_id(url):
    """Extrai o ID do vídeo a partir da URL do YouTube"""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_transcript_youtube(video_id, lang="pt"):
    """Tenta baixar a transcrição nativa do YouTube"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        return transcript
    except:
        return None

def download_audio(video_url):
    """Baixa apenas o áudio do vídeo do YouTube"""
    yt = YouTube(video_url)
    stream = yt.streams.filter(only_audio=True).first()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    stream.download(filename=temp_file.name)
    return temp_file.name

def transcribe_whisper_api(audio_path):
    """Transcreve áudio usando Whisper API (nuvem)"""
    with open(audio_path, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )
    return transcript["segments"]

def transcript_to_text(transcript):
    """Converte lista de blocos em texto contínuo"""
    return " ".join([f"{t['start']:.2f}s: {t['text']}" for t in transcript])

def generate_script_with_gpt(transcription):
    """Envia a transcrição para o GPT reescrever segundo o modelo"""
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
    """Exporta o roteiro em DOCX"""
    doc = Document()
    doc.add_paragraph(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(temp_file.name)
    return temp_file.name

def export_pdf(text):
    """Exporta o roteiro em PDF"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name)
    styles = getSampleStyleSheet()
    story = [Paragraph(p, styles["Normal"]) for p in text.split("\n")]
    for s in story: s.spaceAfter = 12
    doc.build(story)
    return temp_file.name

def export_srt(transcript):
    """Exporta a transcrição com tempo em formato SRT"""
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
    """Formata tempo para SRT (hh:mm:ss,ms)"""
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

# -------- INTERFACE STREAMLIT --------

st.set_page_config(page_title="Agente de Roteiros Virais", layout="wide")
st.title("🎬 Agente de Roteiros Virais")

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
