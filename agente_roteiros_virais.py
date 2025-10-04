import os
import tempfile
import streamlit as st
import yt_dlp
import whisper
from openai import OpenAI

# =========================
# Inicialização do cliente
# =========================
def init_openai():
    openai_key = st.text_input("🔑 Insira sua chave da OpenAI:", type="password")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        return OpenAI(api_key=openai_key)
    return None

# =========================
# Download YouTube (áudio)
# =========================
def download_youtube_audio(url):
    temp_dir = tempfile.mkdtemp()
    out_path = os.path.join(temp_dir, "audio.mp3")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_path,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return out_path
    except Exception as e:
        st.error(f"Erro ao baixar áudio do YouTube: {e}")
        return None

# =========================
# Transcrição via OpenAI API
# =========================
def transcribe_openai(client, file_path):
    try:
        with open(file_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f
            )
        return transcript.text
    except Exception as e:
        st.warning(f"Falha na API da OpenAI. Caindo para Whisper local. Erro: {e}")
        return None

# =========================
# Transcrição via Whisper local
# =========================
def transcribe_whisper_local(file_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        st.warning(f"Erro no Whisper local. Tentando fallback final... {e}")
        return None

# =========================
# Fallback com MoviePy
# =========================
def extract_audio_moviepy(file_path):
    try:
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(file_path)
        audio_path = tempfile.mktemp(suffix=".wav")
        video.audio.write_audiofile(audio_path, logger=None)
        return audio_path
    except Exception as e:
        st.error(f"Erro até no fallback MoviePy: {e}")
        return None

# =========================
# Geração do roteiro viral
# =========================
def gerar_roteiro(client, transcricao):
    prompt = f"""
    Você é um roteirista especializado em vídeos virais para YouTube Shorts.
    Reescreva a transcrição abaixo em um ROTEIRO VIRAL estruturado.

    Estrutura obrigatória:
    - Gancho inicial
    - Contexto/questionamento (30s, deve incluir trechos REESCRITOS da história original para corroborar)
    - Alternância de opostos
    - Resposta inesperada
    - Opinião final
    - CTA
    Além disso: título, sugestão de thumbnail, sugestões de edição.

    Transcrição original:
    {transcricao}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um roteirista criativo especialista em narrativas virais."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao gerar roteiro: {e}")
        return None

# =========================
# App principal Streamlit
# =========================
def main():
    st.title("🎬 Agente de Roteiros Virais")
    st.write("Gere roteiros virais a partir de vídeos do YouTube ou upload manual.")

    client = init_openai()
    if not client:
        st.warning("Insira sua chave da OpenAI para continuar.")
        return

    option = st.radio("Escolha a fonte:", ["YouTube Link", "Upload Manual"])

    transcricao = None
    if option == "YouTube Link":
        url = st.text_input("Cole o link do YouTube:")
        if st.button("Transcrever do YouTube"):
            audio_file = download_youtube_audio(url)
            if audio_file:
                transcricao = transcribe_openai(client, audio_file)
                if not transcricao:
                    transcricao = transcribe_whisper_local(audio_file)

    elif option == "Upload Manual":
        uploaded_file = st.file_uploader("Faça upload de um vídeo/áudio", type=["mp4", "mp3", "wav"])
        if uploaded_file:
            temp_path = os.path.join(tempfile.mkdtemp(), uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            if st.button("Transcrever Upload"):
                transcricao = transcribe_openai(client, temp_path)
                if not transcricao:
                    transcricao = transcribe_whisper_local(temp_path)
                if not transcricao:  # fallback extra
                    extracted_audio = extract_audio_moviepy(temp_path)
                    if extracted_audio:
                        transcricao = transcribe_whisper_local(extracted_audio)

    if transcricao:
        st.subheader("📜 Transcrição")
        st.write(transcricao)

        roteiro = gerar_roteiro(client, transcricao)
        if roteiro:
            st.subheader("🎯 Roteiro Viral")
            st.write(roteiro)

if __name__ == "__main__":
    main()
