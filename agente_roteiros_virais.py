import streamlit as st
import tempfile
import os
from openai import OpenAI

# ===============================
# Configuração Inicial
# ===============================
st.set_page_config(page_title="Agente de Roteiros Virais 🎬", page_icon="🎬", layout="centered")

st.title("🎬 Agente de Roteiros Virais")
st.markdown("Transforme qualquer vídeo em um **roteiro viral estruturado**, com storytelling, ritmo e emoção.")

api_key = st.text_input("🔑 Sua OpenAI API Key:", type="password")

uploaded_file = st.file_uploader("🎥 Faça upload do vídeo (MP4, MOV, MKV, etc.)", type=["mp4", "mov", "mkv"])
fidelity = st.select_slider("🎚️ Nível de fidelidade à transcrição:",
                            options=["Criativo", "Equilibrado", "Fiel"],
                            value="Equilibrado")

# ===============================
# Transcrição com Whisper API
# ===============================
def transcribe_with_openai(video_path, api_key):
    client = OpenAI(api_key=api_key)
    with open(video_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text.strip()

# ===============================
# Geração do Roteiro Viral
# ===============================
def gerar_roteiro(transcricao, fidelidade, api_key):
    client = OpenAI(api_key=api_key)

    prompt = f"""
Você é um roteirista especialista em vídeos curtos virais.  
Crie um roteiro envolvente, emocional e cronologicamente coerente baseado na transcrição a seguir.

Siga esta estrutura narrativa:

🎬 **Início**
1. Primeiros 5 segundos — uma frase que reflita a **thumb** (impactante e chamativa).
2. Até 30 segundos de **contexto e questionamento** que despertem curiosidade.

🎭 **Meio**
- Divida o restante da transcrição em **blocos de até 90 segundos** cada.
- Cada bloco deve alternar entre:
  a) **Momento de tensão, curiosidade ou oposição**.
  b) **Resposta ou superação inesperada**.
- Mantenha **nomes, datas, fatos e detalhes originais** com fidelidade.  
- Cada bloco deve respeitar a **ordem cronológica** dos eventos.

🎁 **Fim**
1. **Recompensa emocional** ou conclusão inspiradora.
2. **Chamada para ação (CTA)** — incentive o público a curtir e seguir.

💡 Nível de fidelidade pedido: {fidelidade}

Transcrição original:
{transcricao}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um roteirista criativo, detalhista e especialista em storytelling viral."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )

    roteiro = response.choices[0].message.content.strip()
    return roteiro


# ===============================
# Execução Principal
# ===============================
if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    progress_bar = st.progress(0)
    progress_text = st.empty()

    try:
        progress_text.text("⏳ Etapa 1/2: Transcrevendo o vídeo com Whisper API...")
        transcricao = transcribe_with_openai(video_path, api_key)
        progress_bar.progress(50)

        progress_text.text("💡 Etapa 2/2: Gerando roteiro viral...")
        roteiro = gerar_roteiro(transcricao, fidelity, api_key)
        progress_bar.progress(100)

        st.success("✅ Roteiro gerado com sucesso!")
        st.subheader("📜 Roteiro Final")
        st.write(roteiro)

        st.download_button("📥 Baixar roteiro em .txt", roteiro, file_name="roteiro_viral.txt")

    except Exception as e:
        st.error(f"Erro: {e}")

    finally:
        os.remove(video_path)
else:
    st.info("👆 Faça upload de um vídeo e insira sua chave da OpenAI para começar.")
