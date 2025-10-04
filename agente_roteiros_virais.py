import os
import shutil
import tempfile
import streamlit as st
import openai
import whisper
import torch
import imageio_ffmpeg as ffmpeg
import time

# ==========================================
# Garante que o Whisper encontre o ffmpeg
# ==========================================
ffmpeg_path = ffmpeg.get_ffmpeg_exe()
shutil.copy(ffmpeg_path, "/tmp/ffmpeg")  # cria um binário no /tmp
os.environ["PATH"] = "/tmp:" + os.environ["PATH"]


# ==========================================
# Funções auxiliares
# ==========================================

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
st.write("Faça upload de um vídeo ou áudio para gerar um roteiro no formato viral.")

api_key = st.text_input("🔑 Digite sua chave da OpenAI:", type="password")
uploaded_file = st.file_uploader("📤 Upload de arquivo de vídeo/áudio", type=["mp4", "mp3", "wav", "m4a"])

if st.button("Gerar Roteiro"):
    if not api_key:
        st.error("Por favor, insira sua chave da OpenAI.")
    elif uploaded_file is None:
        st.error("Por favor, faça upload de um vídeo ou áudio.")
    else:
        # Salva o arquivo temporário
        st.info("📤 Preparando arquivo enviado...")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        audio_path = temp_file.name

        # Barra de progresso para transcrição
        progress_text = "🎙️ Transcrevendo áudio..."
        progress_bar = st.progress(0, text=progress_text)

        transcript = transcribe_whisper_api(audio_path, api_key)
        progress_bar.progress(50, text="⚡ Processando com Whisper...")

        if not transcript:
            transcript = transcribe_whisper_local(audio_path)

        progress_bar.progress(100, text="✅ Transcrição concluída!")

        if transcript:
            # Barra de progresso para geração do roteiro
            roteiro_bar = st.progress(0, text="📝 Criando roteiro viral...")
            time.sleep(1)
            roteiro_bar.progress(30, text="🔍 Analisando transcrição...")
            time.sleep(1)
            roteiro_bar.progress(60, text="🎯 Estruturando roteiro...")
            time.sleep(1)

            roteiro = gerar_roteiro(transcript, api_key)
            roteiro_bar.progress(100, text="✅ Roteiro gerado com sucesso!")

            # Exibe resultados
            st.markdown("### 📜 Transcrição")
            st.write(transcript)

            st.markdown("### 🎯 Roteiro Viral")
            st.write(roteiro)
        else:
            st.error("❌ Não foi possível transcrever o vídeo.")
