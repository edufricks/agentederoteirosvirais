import os
import tempfile
import shutil
import streamlit as st
import openai
import whisper
import torch
import time
import imageio_ffmpeg as ffmpeg

# ==========================================
# Corrige problema do ffmpeg para Whisper local
# ==========================================
ffmpeg_path = ffmpeg.get_ffmpeg_exe()
shutil.copy(ffmpeg_path, "/tmp/ffmpeg")
os.environ["PATH"] = "/tmp:" + os.environ["PATH"]

# ==========================================
# Funções de Transcrição
# ==========================================

def transcribe_with_openai(audio_path, api_key):
    """Transcreve usando a API da OpenAI."""
    openai.api_key = api_key
    try:
        with open(audio_path, "rb") as f:
            client = openai.OpenAI(api_key=api_key)
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return transcript
    except Exception as e:
        st.warning(f"⚠️ Falha na API da OpenAI. Usando Whisper Local. Erro: {e}")
        return None


def transcribe_whisper_local(audio_path):
    """Transcreve com modelo Whisper local."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", device=device)
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"❌ Erro no Whisper local: {e}")
        return None

# ==========================================
# Função de geração de roteiro
# ==========================================

def gerar_roteiro(transcricao, api_key, fidelidade):
    client = openai.OpenAI(api_key=api_key)

    if fidelidade == "Alta":
        nivel_prompt = (
            "Mantenha todos os nomes, eventos, datas e detalhes factuais reais. "
            "Priorize fidelidade total à transcrição original. "
            "Use frases curtas e fortes, mas preserve cada fato relevante."
        )
    elif fidelidade == "Média":
        nivel_prompt = (
            "Mantenha a maioria dos nomes e eventos reais, mas ajuste a narrativa "
            "para ficar fluida e natural em vídeo curto, sem inventar fatos novos."
        )
    else:
        nivel_prompt = (
            "Resuma e simplifique as histórias, mantendo apenas o essencial para o impacto emocional."
        )

    prompt = f"""
Você é um roteirista especialista em vídeos virais.
Sua missão é transformar a transcrição abaixo em um roteiro envolvente e cronológico,
seguindo o ritmo natural do vídeo e cobrindo todas as histórias narradas.

⚙️ Diretrizes principais:
1. Mantenha a ordem cronológica dos fatos.
2. Use pausas naturais e transições suaves a cada mudança de tema.
3. Divida o texto em ritmo de 90 segundos por bloco de fala.
4. Em cada bloco, explore:
   - Contrastes fortes (ex: derrota → superação, perda → vitória)
   - Curiosidade e reviravolta emocional
5. Ao final, inclua:
   - Opinião ou reflexão final (recompensa emocional)
   - CTA para seguir ou curtir

🎯 Fidelidade de conteúdo: {fidelidade}
{nivel_prompt}

📜 Estrutura esperada:
- Título chamativo
- Ideia de thumb (imagem + texto)
- Blocos de roteiro cronológicos (com base no áudio)
- Sugestões para Shorts (3)
- Sugestões de edição (3)

🗣️ Transcrição original:
\"\"\"{transcricao}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Erro ao gerar roteiro: {e}")
        return None

# ==========================================
# INTERFACE STREAMLIT
# ==========================================

st.set_page_config(page_title="Agente de Roteiros Virais", page_icon="🎬", layout="centered")
st.title("🎬 Agente de Roteiros Virais")
st.write("Envie um vídeo ou áudio para gerar automaticamente um roteiro viral com base na transcrição.")

api_key = st.text_input("🔑 Digite sua chave da OpenAI:", type="password")

uploaded_file = st.file_uploader(
    "📤 Envie um arquivo de vídeo ou áudio",
    type=["mp4", "mp3", "wav", "m4a", "mov"]
)

fidelidade = st.radio(
    "🎯 Nível de fidelidade ao conteúdo original:",
    ["Alta", "Média", "Baixa"],
    index=0,
    horizontal=True
)

if st.button("🚀 Gerar Roteiro"):
    if not api_key:
        st.error("Por favor, insira sua chave da OpenAI.")
    elif not uploaded_file:
        st.error("Envie um arquivo de vídeo ou áudio para continuar.")
    else:
        with st.spinner("⏳ Processando arquivo..."):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name
            time.sleep(1)

        progress_text = "🎙️ Transcrevendo áudio..."
        progress_bar = st.progress(0, text=progress_text)

        transcript = transcribe_with_openai(video_path, api_key)
        progress_bar.progress(50, text="🎧 Quase lá...")

        if not transcript:
            transcript = transcribe_whisper_local(video_path)

        progress_bar.progress(100, text="✅ Transcrição concluída!")

        if transcript:
            st.subheader("🗒️ Transcrição")
            st.write(transcript)

            with st.spinner("📝 Gerando roteiro viral..."):
                roteiro = gerar_roteiro(transcript, api_key, fidelidade)

            if roteiro:
                st.success("✅ Roteiro gerado com sucesso!")
                st.markdown("### 🎬 Roteiro Final")
                st.write(roteiro)
            else:
                st.error("❌ Não foi possível gerar o roteiro.")
        else:
            st.error("❌ Não foi possível obter a transcrição do vídeo.")
