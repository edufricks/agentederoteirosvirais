import os
import shutil
import streamlit as st
import tempfile
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
    """Gera o roteiro final no formato viral respeitando a cronologia e os fatos."""
    openai.api_key = api_key

    prompt = f"""
Você é um roteirista especialista em vídeos virais com alta retenção.
Sua missão é transformar a transcrição abaixo em um roteiro no formato viral, **sem perder nenhum detalhe real** e **mantendo a ordem cronológica**.

🎯 OBJETIVO:
Criar um roteiro que conte todas as histórias e informações da transcrição de forma envolvente, emocional e cinematográfica — mas sem alterar ou omitir fatos, nomes, números, espécies, locais, datas ou qualquer dado real.

⚠️ REGRAS OBRIGATÓRIAS:
1. **Todos os dados reais da transcrição devem aparecer no roteiro.**
   - Inclua nomes, números, locais, datas, espécies, medidas, termos científicos, curiosidades e comparações.
   - Não simplifique nem generalize fatos (ex: se disser “Ochotona, gênero de mamíferos da família Ochotonidae”, mantenha exatamente isso no roteiro).
2. **Não invente fatos.**
   - Pode melhorar a forma de contar, mas nunca criar informações novas.
3. **Respeite a ordem cronológica do vídeo original.**
4. **Estilo narrativo:** linguagem natural, fluida e emocional, como em vídeos documentais virais ou narrativas do YouTube.
5. **Ritmo:** frases curtas, interrogações, pausas dramáticas e ganchos a cada 20–30 segundos.
6. **Estrutura sugerida:**

Início:
   - 5 segundos que reflitam a thumb (impacto e curiosidade)
   - Até 30 segundos de contexto e questionamento inicial

Meio (pode conter vários blocos, até cobrir todas as histórias):
   - Cada bloco (até 90 segundos) deve:
       a) Alternar entre momentos opostos (ex: descoberta vs dúvida, sucesso vs fracasso, fragilidade vs superação)
       b) Fechar com uma resposta surpreendente, insight ou virada
   - Continue criando novos blocos até representar todo o conteúdo da transcrição

Fim:
   - Recompensa final: opinião ou conclusão emocional sobre a jornada
   - CTA de engajamento (seguir, curtir, comentar, etc.)

7. **No final do roteiro, adicione também:**
   - 🎬 **Título chamativo**
   - 🖼️ **Ideia de Thumb (imagem + texto)**
   - 🎞️ **3 ideias de Shorts**
   - ✂️ **3 sugestões de edição (efeitos, cortes, transições)**

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
st.write("Faça upload de um vídeo ou áudio para gerar um roteiro fiel e envolvente no formato viral.")

api_key = st.text_input("🔑 Digite sua chave da OpenAI:", type="password")
uploaded_file = st.file_uploader("📤 Upload de vídeo/áudio", type=["mp4", "mp3", "wav", "m4a"])

if st.button("Gerar Roteiro"):
    if not api_key:
        st.error("Por favor, insira sua chave da OpenAI.")
    else:
        transcript = None
        audio_path = None

        if uploaded_file is not None:
            st.info("📤 Usando arquivo enviado pelo usuário...")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            audio_path = temp_file.name

        if audio_path:
            progress_bar = st.progress(0)
            with st.spinner("🎙️ Transcrevendo áudio..."):
                transcript = transcribe_whisper_api(audio_path, api_key)
                progress_bar.progress(50)
                if not transcript:
                    transcript = transcribe_whisper_local(audio_path)

            if transcript:
                with st.spinner("📝 Gerando roteiro..."):
                    roteiro = gerar_roteiro(transcript, api_key)
                    progress_bar.progress(100)

                st.success("✅ Roteiro gerado com sucesso!")
                st.markdown("### 📜 Transcrição")
                st.write(transcript)

                st.markdown("### 🎯 Roteiro Viral")
                st.write(roteiro)
            else:
                st.error("❌ Não foi possível obter transcrição do vídeo.")
