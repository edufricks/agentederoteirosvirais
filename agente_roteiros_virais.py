import os
import streamlit as st
import tempfile
from openai import OpenAI

# -------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------

def transcribe_with_openai(audio_path, api_key):
    """Transcreve o áudio usando apenas a API da OpenAI"""
    st.info("🔊 Transcrevendo áudio com Whisper API...")

    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

def generate_viral_script(transcription_text, api_key):
    """Cria roteiro viral estruturado a partir da transcrição"""
    st.info("✍️ Criando roteiro viral com GPT...")

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

# -------------------------------
# APP STREAMLIT
# -------------------------------

def main():
    st.title("🎬 Agente de Roteiros Virais (Upload Manual)")

    api_key = st.text_input("🔑 Insira sua OpenAI API Key:", type="password")

    uploaded_file = st.file_uploader("📂 Faça upload de um vídeo (MP4, MOV, AVI, MP3)...", type=["mp4", "mov", "avi", "mp3"])

    if uploaded_file and api_key:
        with st.spinner("Processando vídeo..."):
            # Barra de progresso
            progress = st.progress(0)

            # Salvar arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                tmpfile.write(uploaded_file.read())
                video_path = tmpfile.name
            progress.progress(30)

            # Transcrever com Whisper API
            transcription_text = transcribe_with_openai(video_path, api_key)
            progress.progress(70)

            st.subheader("📜 Transcrição")
            st.text_area("", transcription_text, height=200)

            # Gerar roteiro viral
            script = generate_viral_script(transcription_text, api_key)
            progress.progress(100)

            st.subheader("🎯 Roteiro Viral")
            st.write(script)

if __name__ == "__main__":
    main()
