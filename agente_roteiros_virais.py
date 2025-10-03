import os
import shutil
import streamlit as st
import tempfile
import yt_dlp
import openai
import whisper
import torch
import imageio_ffmpeg as ffmpeg

# ==========================================
# Garante que o Whisper encontre o ffmpeg
# ==========================================
ffmpeg_path = ffmpeg.get_ffmpeg_exe()
shutil.copy(ffmpeg_path, "/tmp/ffmpeg")  # cria um binário no /tmp
os.environ["PATH"] = "/tmp:" + os.environ["PATH"]


# ==========================================
# Funções auxiliares
# ==========================================

def download_audio(url: str) -> str:
    """Baixa apenas o áudio do YouTube e retorna o caminho do arquivo local."""
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = ydl.prepare_filename(info)

    return audio_path


def transcribe_whisper_api(audio_path: str, api_key: str):
    """Transcreve com Whisper API da OpenAI."""
    openai.api_key = api_key
    try:
        with open(audio_path, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return transcript
    except Exception as e:
        st.warning(f"⚠️ Falha na API da OpenAI. Usando Whisper Local. Erro: {e}")
        return None


def transcribe_whisper_local(audio_path: str):
    """Transcreve com Whisper rodando localmente."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    result = model.transcribe(audio_path)
    return result["text"]


def gerar_roteiro(transcricao: str, api_key: str):
    """Gera o roteiro final no formato viral."""
    openai.api_key = api_key

    prompt = f"""
Você é um roteirista especialista em vídeos virais. 
Transforme a transcrição abaixo em um roteiro no formato viral, seguindo exatamente esta estrutura:

Roteiro no formato viral:
 - Gancho inicial
 - Contexto/questionamento
 - Alternância de opostos
 - Resposta inesperada
 - Opinião final
 - CTA
Além de: título, thumb, shorts, sugestões de edição

Transcrição original:
\"\"\"{transcricao}\"\"\"
"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content


# ==========================================
# Streamlit App
# ==========================================

st.title("🎬 Agente de Roteiros Virais")
st.write("Cole o link de um vídeo do YouTube **ou faça upload** para gerar um roteiro no formato viral.")

api_key = st.text_input("🔑 Digite sua chave da OpenAI:", type="password")
url = st.text_input("📺 URL do vídeo do YouTube:")
uploaded_file = st.file_uploader("📤 Ou faça upload de um arquivo de vídeo/áudio", type=["mp4", "mp3", "wav", "m4a"])

if st.button("Gerar Roteiro"):
    if not api_key:
        st.error("Por favor, insira sua chave da OpenAI.")
    else:
        transcript = None
        audio_path = None

        if url:
            try:
                st.info("📥 Baixando áudio do YouTube...")
                audio_path = download_audio(url)
            except Exception as e:
                st.warning(f"⚠️ Falha ao baixar do YouTube. Erro: {e}")

        if not audio_path and uploaded_file is not None:
            st.info("📤 Usando arquivo enviado pelo usuário...")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            audio_path = temp_file.name

        if audio_path:
            with st.spinner("🎙️ Transcrevendo áudio..."):
                transcript = transcribe_whisper_api(audio_path, api_key)
                if not transcript:
                    transcript = transcribe_whisper_local(audio_path)

        if transcript:
            with st.spinner("📝 Gerando roteiro..."):
                roteiro = gerar_roteiro(transcript, api_key)

            st.success("✅ Roteiro gerado com sucesso!")
            st.markdown("### 📜 Transcrição")
            st.write(transcript)

            st.markdown("### 🎯 Roteiro Viral")
            st.write(roteiro)
        else:
            st.error("❌ Não foi possível obter transcrição do vídeo.")
