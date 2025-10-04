import os
import streamlit as st
from moviepy.editor import VideoFileClip
import tempfile
import torch
import whisper
from openai import OpenAI

# -------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------

def extract_audio(video_path):
    """Extrai áudio de um vídeo e retorna caminho do arquivo de áudio"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        audio_path = tmpfile.name
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec="mp3")
    return audio_path

def transcribe_audio(audio_path, api_key):
    """Tenta transcrever usando API da OpenAI, se falhar cai para Whisper local"""
    st.info("🔊 Transcrevendo áudio...")

    # Tentativa 1 - API OpenAI Whisper
    try:
        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcript.text
    except Exception as e:
        st.warning(f"Falha na API da OpenAI. Caindo para Whisper local. Erro: {e}")

    # Tentativa 2 - Whisper local
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Erro no Whisper local: {e}")
        return None

def generate_viral_script(transcription_text, api_key):
    """Cria roteiro viral estruturado a partir da transcrição"""
    st.info("✍️ Criando roteiro viral...")

    prompt = f"""
Você é um roteirista de vídeos virais para YouTube. 
Transforme a transcrição abaixo em um roteiro VIRAL, obedecendo exatamente esta estrutura:

Título
Thumb
Roteiro:
- Gancho Inicial
- Contexto/Questionamento (30s) → inclua histórias REESCRITAS a partir da transcrição
- Alternância de Opostos
- Resposta Inesperada
- Opinião Final
- CTA
Extras:
- 3 ideias para Shorts/TikToks
- 3 sugestões concretas de edição

IMPORTANTE:
- Use os fatos e histórias da transcrição como base (refraseie, mas não invente).
- Sempre evidencie trechos originais (ex: jogos, frases ditas, críticas).
- Seja impactante e conciso.

Transcrição:
\"\"\"{transcription_text}\"\"\"
"""

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um roteirista profissional de vídeos virais."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=1800
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[Erro gerando roteiro: {e}]"

# -------------------------------
# APP STREAMLIT
# -------------------------------

def main():
    st.title("🎬 Agente de Roteiros Virais (Upload Manual)")

    api_key = st.text_input("🔑 Insira sua OpenAI API Key:", type="password")

    uploaded_file = st.file_uploader("📂 Faça upload de um vídeo (MP4, MOV, AVI)...", type=["mp4", "mov", "avi"])

    if uploaded_file and api_key:
        with st.spinner("Processando vídeo..."):
            # Barra de progresso (estética)
            progress = st.progress(0)

            # Salvar arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                tmpfile.write(uploaded_file.read())
                video_path = tmpfile.name
            progress.progress(20)

            # Extrair áudio
            audio_path = extract_audio(video_path)
            progress.progress(40)

            # Transcrever
            transcription_text = transcribe_audio(audio_path, api_key)
            progress.progress(70)

            if transcription_text:
                st.subheader("📜 Transcrição")
                st.text_area("", transcription_text, height=200)

                # Gerar roteiro viral
                script = generate_viral_script(transcription_text, api_key)
                progress.progress(100)

                st.subheader("🎯 Roteiro Viral")
                st.write(script)
            else:
                st.error("Não foi possível transcrever o áudio.")

if __name__ == "__main__":
    main()
