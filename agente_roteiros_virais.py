import streamlit as st
import tempfile
import os
from tqdm import tqdm
from faster_whisper import WhisperModel
from openai import OpenAI

# ===============================
# Configuração Inicial
# ===============================
st.set_page_config(page_title="Agente de Roteiros Virais 🎬", page_icon="🎬", layout="centered")

st.title("🎬 Agente de Roteiros Virais")
st.markdown("Transforme qualquer vídeo em um **roteiro viral**, seguindo uma estrutura narrativa envolvente e emocional.")

api_key = st.text_input("🔑 Sua OpenAI API Key:", type="password")
fidelity = st.select_slider("🎚️ Nível de fidelidade à transcrição:",
                            options=["Criativo", "Equilibrado", "Fiel"],
                            value="Equilibrado")

uploaded_file = st.file_uploader("🎥 Faça upload do vídeo (MP4, MOV, MKV, etc.)", type=["mp4", "mov", "mkv"])

# ===============================
# Transcrição com Faster Whisper
# ===============================
def transcribe_local(video_path):
    try:
        st.info("🎧 Transcrevendo o áudio localmente com **Faster Whisper**...")
        model = WhisperModel("small", device="cpu", compute_type="int8")

        segments, info = model.transcribe(video_path, beam_size=5)
        text = " ".join(segment.text for segment in segments)
        return text.strip()
    except Exception as e:
        st.error(f"Erro durante a transcrição: {e}")
        return None

# ===============================
# Geração do roteiro
# ===============================
def gerar_roteiro(transcricao, fidelidade, api_key):
    try:
        client = OpenAI(api_key=api_key)
        st.info("🧠 Gerando roteiro viral com GPT... isso pode levar alguns segundos.")

        prompt = f"""
Você é um roteirista especialista em narrativas virais de vídeos curtos.
Use a transcrição abaixo para criar um roteiro completo, fiel aos fatos e aos nomes mencionados, mas com ritmo e emoção.

O roteiro deve seguir esta estrutura:

🟢 **Início**
1. Uma frase de até 5 segundos que reflita o impacto visual da thumb.
2. Até 30 segundos de contexto com uma pergunta ou conflito.

🟡 **Meio**
- Divida o restante da transcrição em blocos de até 90 segundos cada.
- Cada bloco deve alternar entre:
  a) Momento de tensão, curiosidade ou oposição.
  b) Resposta ou superação inesperada.
- Mantenha todos os nomes e dados originais da transcrição.

🔵 **Fim**
1. Recompensa emocional ou conclusão.
2. Chamada para ação: peça para seguir e curtir.

A fidelidade ao texto deve ser: {fidelidade}.
Aqui está a transcrição completa:

{transcricao}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um roteirista criativo e detalhista, especialista em storytelling viral."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
        )

        roteiro = response.choices[0].message.content.strip()
        return roteiro
    except Exception as e:
        st.error(f"Erro ao gerar roteiro: {e}")
        return None

# ===============================
# Execução principal
# ===============================
if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    progress_bar = st.progress(0)
    progress_text = st.empty()

    progress_text.text("⏳ Etapa 1/2: Transcrevendo o vídeo...")
    transcricao = transcribe_local(video_path)
    progress_bar.progress(50)

    if transcricao:
        progress_text.text("💡 Etapa 2/2: Criando roteiro viral...")
        roteiro = gerar_roteiro(transcricao, fidelity, api_key)
        progress_bar.progress(100)

        if roteiro:
            st.success("✅ Roteiro viral gerado com sucesso!")
            st.subheader("🎬 Roteiro Final")
            st.write(roteiro)
        else:
            st.error("❌ Não foi possível gerar o roteiro.")
    else:
        st.error("❌ Falha ao transcrever o vídeo.")

    os.remove(video_path)
else:
    st.info("👆 Faça upload de um vídeo e insira sua chave da OpenAI para começar.")
