import os
import shutil
import streamlit as st
import tempfile
import openai
import whisper
import torch
import imageio_ffmpeg as ffmpeg

# ==========================================
# Configura ffmpeg para Whisper local
# ==========================================
ffmpeg_path = ffmpeg.get_ffmpeg_exe()
shutil.copy(ffmpeg_path, "/tmp/ffmpeg")
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


def gerar_prompt(transcricao: str, fidelidade: str):
    """Gera o prompt de acordo com o nível de fidelidade selecionado."""

    base_prompt = f"""
Você é um roteirista especialista em vídeos virais com alta retenção.
Sua missão é transformar a transcrição abaixo em um roteiro no formato viral, mantendo a ordem cronológica e o ritmo envolvente.

🎯 OBJETIVO:
Criar um roteiro que conte todas as histórias e informações da transcrição de forma emocional e cinematográfica — respeitando os fatos e mantendo o interesse do público até o final.

⚙️ Nível de fidelidade: {fidelidade}

"""

    if fidelidade == "Alta fidelidade (máxima precisão factual)":
        detalhes = """
Regras específicas:
1. Nenhum dado pode ser omitido — inclua todos os nomes, números, locais, espécies, medidas, curiosidades e termos originais.
2. Preserve 100% da veracidade factual.
3. Reescreva com fluidez, mas nunca resuma ou simplifique termos.
4. Respeite rigorosamente a ordem cronológica da fala.
5. Transforme os fatos em narrativa envolvente e emocional, sem cortar nenhuma história.
"""
    elif fidelidade == "Equilibrada (entre precisão e narrativa)":
        detalhes = """
Regras específicas:
1. Mantenha todos os dados relevantes, mas com foco em fluidez e ritmo.
2. Você pode condensar trechos longos mantendo o sentido e os principais fatos.
3. Use ganchos, pausas e curiosidades para manter atenção.
4. Preserve a ordem cronológica e os fatos mais importantes.
"""
    else:  # Criativa
        detalhes = """
Regras específicas:
1. Use os fatos como base, mas pode reescrever criativamente trechos pouco claros.
2. Mantenha o espírito e a essência de cada história, mesmo que o texto seja reorganizado.
3. Crie ritmo e emoção com liberdade estilística, mas sem inventar eventos inexistentes.
"""

    estrutura = """
🧩 Estrutura obrigatória:

Início:
   - 5 segundos que reflitam a thumb (impacto e curiosidade)
   - Até 30 segundos de contexto e questionamento inicial

Meio (quantos blocos forem necessários até cobrir todas as histórias):
   - Cada bloco (até 90 segundos) deve:
       a) Alternar entre momentos opostos (ex: dúvida vs conquista)
       b) Fechar com uma resposta ou virada inesperada

Fim:
   - Recompensa final: opinião ou reflexão
   - CTA de engajamento (seguir, curtir, comentar)

🪄 Linguagem:
- Frases curtas e diretas.
- Interrogações e pausas estratégicas.
- Emoção e ritmo natural, como narrativas virais do YouTube.

No final, adicione:
- 🎬 Título chamativo
- 🖼️ Ideia de Thumb (imagem + texto)
- 🎞️ 3 ideias de Shorts
- ✂️ 3 sugestões de edição (efeitos, cortes, transições)

Transcrição original:
\"\"\"{transcricao}\"\"\"
"""

    return base_prompt + detalhes + estrutura


def gerar_roteiro(transcricao: str, api_key: str, fidelidade: str):
    """Gera o roteiro final no formato viral."""
    openai.api_key = api_key
    prompt = gerar_prompt(transcricao, fidelidade)

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content


# ==========================================
# Streamlit App
# ==========================================

st.title("🎬 Agente de Roteiros Virais 2.1")
st.write("Faça upload de um vídeo ou áudio para gerar um roteiro fiel e envolvente no formato viral.")

api_key = st.text_input("🔑 Digite sua chave da OpenAI:", type="password")
uploaded_file = st.file_uploader("📤 Upload de vídeo/áudio", type=["mp4", "mp3", "wav", "m4a"])

fidelidade = st.selectbox(
    "🎯 Nível de fidelidade do roteiro:",
    [
        "Alta fidelidade (máxima precisão factual)",
        "Equilibrada (entre precisão e narrativa)",
        "Criativa (ênfase na emoção e ritmo)"
    ],
    index=1
)

if st.button("Gerar Roteiro"):
    if not api_key:
        st.error("Por favor, insira sua chave da OpenAI.")
    elif not uploaded_file:
        st.error("Envie um arquivo de vídeo ou áudio para continuar.")
    else:
        transcript = None
        progress_bar = st.progress(0)

        st.info("📤 Processando arquivo enviado...")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        audio_path = temp_file.name

        with st.spinner("🎙️ Transcrevendo áudio..."):
            transcript = transcribe_whisper_api(audio_path, api_key)
            progress_bar.progress(50)
            if not transcript:
                transcript = transcribe_whisper_local(audio_path)

        if transcript:
            with st.spinner(f"📝 Gerando roteiro ({fidelidade.lower()})..."):
                roteiro = gerar_roteiro(transcript, api_key, fidelidade)
                progress_bar.progress(100)

            st.success("✅ Roteiro gerado com sucesso!")
            st.markdown("### 📜 Transcrição")
            st.write(transcript)

            st.markdown("### 🎯 Roteiro Viral")
            st.write(roteiro)
        else:
            st.error("❌ Não foi possível obter transcrição do vídeo.")
