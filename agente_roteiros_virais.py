import os
import streamlit as st
import tempfile
import yt_dlp
import openai
import whisper
from moviepy.editor import VideoFileClip

# -----------------------
# Função para baixar áudio de YouTube
# -----------------------
def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': tempfile.mktemp(suffix=".mp3"),
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

# -----------------------
# Transcrição - Whisper API (cloud)
# -----------------------
def transcribe_whisper_api(file_path, api_key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        with open(file_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return transcript
    except Exception as e:
        st.warning(f"Falha na API da OpenAI. Caindo para Whisper local. Erro: {e}")
        return None

# -----------------------
# Transcrição - Whisper local
# -----------------------
def transcribe_whisper_local(file_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        st.warning(f"Falha no Whisper local. Caindo para MoviePy. Erro: {e}")
        return None

# -----------------------
# Transcrição - MoviePy (Fallback final - extrai só áudio bruto)
# -----------------------
def transcribe_moviepy(file_path):
    try:
        video = VideoFileClip(file_path)
        audio_path = tempfile.mktemp(suffix=".wav")
        video.audio.write_audiofile(audio_path, logger=None)
        return f"[Áudio extraído em {audio_path}, mas sem transcrição automática disponível.]"
    except Exception as e:
        st.error(f"Erro até no fallback MoviePy: {e}")
        return None

# -----------------------
# Gerar roteiro viral
# -----------------------
def gerar_roteiro(transcricao, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt = f"""
Você é um roteirista especialista em vídeos virais. 
Sua missão é transformar a transcrição abaixo em um roteiro no formato viral.

⚠️ Regras obrigatórias:
1. Cada bloco (gancho, contexto/questionamento, alternância de opostos, resposta inesperada, opinião final, CTA) deve conter pelo menos **uma história real da transcrição**, reescrita de forma impactante e envolvente.
2. Não invente fatos. Use nomes, eventos, datas e histórias reais da transcrição. Se algo não estiver claro, reescreva criativamente mas sem criar fatos novos.
3. Mantenha a estrutura **fixa**:
   - Gancho inicial (com impacto e curiosidade)
   - Contexto/questionamento (incluindo pelo menos 1 história real da transcrição)
   - Alternância de opostos (críticas vs conquistas, fracassos vs vitórias — com base no que ocorreu no vídeo)
   - Resposta inesperada (a reviravolta ou lição mais surpreendente — baseada no vídeo)
   - Opinião final (lição inspiradora ou conclusão)
   - CTA (convite para engajamento ou próxima ação)
4. Além do roteiro, gere também:
   - Título chamativo
   - Ideia de Thumb (imagem + texto)
   - Sugestões para Shorts (3 ideias)
   - Sugestões de edição (3 ideias, incluindo cortes e efeitos)

Transcrição original:
\"\"\"{transcricao}\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

# -----------------------
# Interface Streamlit
# -----------------------
st.title("🎬 Agente de Roteiros Virais")
st.write("Transforme qualquer vídeo em um roteiro viral pronto para edição.")

# Entrada da chave da OpenAI
api_key = st.text_input("🔑 Insira sua chave da OpenAI:", type="password")

if api_key:
    opcao = st.radio("Selecione a origem do vídeo:", ["YouTube", "Upload manual"])

    video_path = None
    if opcao == "YouTube":
        url = st.text_input("Cole o link do vídeo do YouTube:")
        if url and st.button("Baixar & Transcrever"):
            with st.spinner("Baixando áudio do YouTube..."):
                video_path = download_audio(url)
    else:
        uploaded_file = st.file_uploader("Faça upload do vídeo:", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_file is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

    if video_path:
        with st.spinner("Transcrevendo áudio..."):
            transcript = transcribe_whisper_api(video_path, api_key)
            if not transcript:
                transcript = transcribe_whisper_local(video_path)
            if not transcript:
                transcript = transcribe_moviepy(video_path)

        if transcript:
            st.subheader("📜 Transcrição")
            st.write(transcript[:2000] + "..." if len(transcript) > 2000 else transcript)

            with st.spinner("Gerando roteiro viral..."):
                roteiro = gerar_roteiro(transcript, api_key)

            st.subheader("🎬 Roteiro Viral")
            st.write(roteiro)
else:
    st.info("Por favor, insira sua chave da OpenAI para começar.")
