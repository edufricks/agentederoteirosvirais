import streamlit as st
import openai
import tempfile
import yt_dlp
import whisper
import time

# -------------------
# Configuração
# -------------------
st.set_page_config(page_title="Agente de Roteiros Virais 🎬", layout="wide")
st.title("🎬 Agente de Roteiros Virais")
st.write("Transforme vídeos do YouTube em **transcrição, roteiro viral estruturado e legendas (.srt)**.")

openai_api_key = st.text_input("🔑 Digite sua chave da OpenAI:", type="password")
if openai_api_key:
    openai.api_key = openai_api_key

# -------------------
# Função para baixar áudio
# -------------------
def download_audio(video_url):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_file.name,
        'quiet': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return temp_file.name

# -------------------
# Transcrição via API OpenAI
# -------------------
def transcribe_whisper_api(audio_path, retries=3):
    for attempt in range(retries):
        try:
            with open(audio_path, "rb") as f:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json"
                )
            return transcript["segments"]
        except openai.RateLimitError:
            wait = (2 ** attempt) * 5
            st.warning(f"⚠️ Rate limit na nuvem. Tentando novamente em {wait}s...")
            time.sleep(wait)
        except Exception as e:
            st.error(f"Erro API Whisper: {e}")
            return None
    return None

# -------------------
# Transcrição via Whisper Local
# -------------------
def transcribe_whisper_local(audio_path):
    st.info("Rodando Whisper local (isso pode levar alguns minutos)...")
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    return result["segments"]

# -------------------
# Pipeline híbrido
# -------------------
def transcribe_audio(audio_path):
    if openai_api_key:  # tenta nuvem primeiro
        segments = transcribe_whisper_api(audio_path)
        if segments:
            return segments
        else:
            st.warning("🔄 Falhou na nuvem. Tentando local...")
            return transcribe_whisper_local(audio_path)
    else:
        st.warning("⚠️ Nenhuma chave da OpenAI. Usando Whisper local direto.")
        return transcribe_whisper_local(audio_path)

# -------------------
# Geração do roteiro estruturado
# -------------------
def gerar_roteiro(transcricao_texto):
    if not openai_api_key:
        st.error("⚠️ Você precisa inserir sua chave da OpenAI para gerar o roteiro estruturado.")
        return None
    
    prompt = f"""
    Você é um roteirista especializado em vídeos virais para YouTube e TikTok.
    Reescreva o roteiro a partir desta transcrição de vídeo:

    {transcricao_texto}

    Estruture seguindo este formato:
    1. **Hook inicial (0-15s)** – frase impactante que prende a atenção.
    2. **Apresentação rápida** – explique quem fala e por que o público deve ouvir.
    3. **Desenvolvimento principal (conteúdo em blocos de 20-40s)** – organize a narrativa em tópicos com sugestões de cortes e recursos visuais.
    4. **Inserções visuais sugeridas** – indique onde incluir imagens, gráficos, memes ou efeitos para manter a retenção.
    5. **Call to Action (final)** – sugestão de CTA claro e envolvente (curtir, comentar, seguir).

    Além disso:
    - Marque sugestões de minutagem aproximada (com base na transcrição).
    - Dê ideias para cortes dinâmicos que ajudem a manter a retenção.
    - Sugira ajustes no tom de voz e ritmo.
    - Se identificar momentos de baixa energia, proponha cortes ou reforços narrativos.

    Entregue em formato bem organizado, pronto para ser usado em edição.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Você é um especialista em roteiros virais para YouTube e TikTok."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao gerar roteiro: {e}")
        return None

# -------------------
# Interface principal
# -------------------
url = st.text_input("📺 Cole o link do vídeo do YouTube:")

if st.button("🚀 Processar vídeo"):
    if url:
        with st.spinner("Baixando áudio..."):
            audio_path = download_audio(url)

        with st.spinner("Transcrevendo..."):
            segments = transcribe_audio(audio_path)

        if segments:
            st.success("✅ Transcrição concluída!")

            # Mostrar transcrição bruta
            transcript_text = "\n".join([f"[{seg['start']:.2f}s] {seg['text']}" for seg in segments])
            st.subheader("📄 Transcrição")
            st.text_area("Transcrição completa:", transcript_text, height=300)

            # Exportar .srt
            srt_content = ""
            for i, seg in enumerate(segments, start=1):
                start_time = time.strftime('%H:%M:%S', time.gmtime(seg['start']))
                end_time = time.strftime('%H:%M:%S', time.gmtime(seg.get('end', seg['start'] + seg['duration'])))
                srt_content += f"{i}\n{start_time},000 --> {end_time},000\n{seg['text']}\n\n"

            st.download_button("⬇️ Baixar legendas (.srt)", srt_content, "transcricao.srt", "text/plain")

            # Gerar roteiro viral estruturado
            with st.spinner("Gerando roteiro estruturado..."):
                roteiro = gerar_roteiro(transcript_text)

            if roteiro:
                st.subheader("🎬 Roteiro Viral Estruturado")
                st.markdown(roteiro)
                st.download_button("⬇️ Baixar roteiro (.txt)", roteiro, "roteiro_viral.txt", "text/plain")
    else:
        st.warning("Insira um link válido do YouTube.")
