import os
import shutil
import tempfile
import time
import streamlit as st
import yt_dlp

# openai + whisper + ffmpeg helper
import openai
try:
    from openai.error import RateLimitError, OpenAIError
except Exception:
    RateLimitError = Exception
    OpenAIError = Exception

import whisper
import torch
import imageio_ffmpeg as ffmpeg

# --------------------------
# Ensure ffmpeg is available (works in Streamlit Cloud & local)
# --------------------------
def ensure_ffmpeg_on_path():
    try:
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        target = "/tmp/ffmpeg"
        if not os.path.exists(target):
            shutil.copy(ffmpeg_exe, target)
            os.chmod(target, 0o755)
        os.environ["PATH"] = "/tmp:" + os.environ.get("PATH", "")
        return True
    except Exception as e:
        # Non-fatal; local machines may have ffmpeg preinstalled
        st.warning(f"Não foi possível garantir ffmpeg via imageio-ffmpeg: {e}")
        return False

# Run once at startup
ensure_ffmpeg_on_path()

# --------------------------
# Helpers: download audio, extract audio (moviepy fallback)
# --------------------------
def download_audio_from_youtube(url):
    temp_dir = tempfile.mkdtemp()
    out_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        return filename
    except Exception as e:
        st.warning(f"yt-dlp falhou ao baixar áudio: {e}")
        return None

def extract_audio_with_moviepy(video_path):
    try:
        from moviepy.editor import VideoFileClip  # optional import
        clip = VideoFileClip(video_path)
        audio_path = tempfile.mktemp(suffix=".wav")
        clip.audio.write_audiofile(audio_path, logger=None)
        return audio_path
    except Exception as e:
        st.warning(f"MoviePy fallback falhou: {e}")
        return None

# --------------------------
# Transcrição: OpenAI Whisper API (returns dict with 'text' and optional 'segments')
# --------------------------
def transcribe_with_openai_api(file_path):
    try:
        with open(file_path, "rb") as f:
            # request verbose_json to try to get segments too (if possible)
            resp = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json"
            )
        # resp may contain 'text' and 'segments'
        text = resp.get("text") if isinstance(resp, dict) else getattr(resp, "text", None)
        segments = resp.get("segments") if isinstance(resp, dict) else None
        return {"success": True, "text": text, "segments": segments}
    except RateLimitError as e:
        return {"success": False, "error": "rate_limit", "detail": str(e)}
    except OpenAIError as e:
        return {"success": False, "error": "openai_error", "detail": str(e)}
    except Exception as e:
        return {"success": False, "error": "unknown", "detail": str(e)}

# --------------------------
# Transcrição: Whisper local
# returns dict with 'text' and 'segments' (segments comes from model.transcribe result)
# --------------------------
def transcribe_with_whisper_local(file_path, model_name="base"):
    try:
        # ensure ffmpeg available
        ensure_ffmpeg_on_path()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_name, device=device)
        result = model.transcribe(file_path)
        text = result.get("text")
        segments = result.get("segments")  # each has start, end, text
        return {"success": True, "text": text, "segments": segments}
    except Exception as e:
        return {"success": False, "error": "whisper_local_failed", "detail": str(e)}

# --------------------------
# SRT exporter (given segments)
# --------------------------
def segments_to_srt(segments):
    def fmt(t):
        millis = int((t - int(t)) * 1000)
        s = int(t)
        hrs = s // 3600
        mins = (s % 3600) // 60
        secs = s % 60
        return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = seg.get("start")
        end = seg.get("end") or (start + seg.get("duration", 0))
        text = seg.get("text").strip()
        lines.append(f"{i}\n{fmt(start)} --> {fmt(end)}\n{text}\n")
    return "\n".join(lines)

# --------------------------
# Roteiro: prompt rígido que obriga o uso das histórias reais por bloco
# --------------------------
def generate_viral_script(transcription_text):
    prompt = f"""
You are an expert viral-video scriptwriter. Transform the transcription below into a VIRAL YOUTUBE script.
REQUIREMENTS (must follow exactly):
1) Output these sections (with headings): Title, Thumbnail idea, Hooks, Script:
   - Script must contain these subsections: Gancho (Hook), Contexto/Questionamento (30s), Alternância de Opostos, Resposta Inesperada, Opinião Final, CTA.
2) IMPORTANT: each subsection MUST include at least ONE real story/event/line FROM THE TRANSCRIPTION, rephrased (do NOT invent new facts).
3) Use names, events and explicit moments from the transcription as evidence. Do not fabricate dates/quotes.
4) Also include 3 short ideas for Shorts/TikToks (timestamps if possible) and 3 concrete editing suggestions.
Transcription:
\"\"\"{transcription_text}\"\"\"
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a helpful, accurate scriptwriter."},
                      {"role":"user","content":prompt}],
            temperature=0.6,
            max_tokens=1800
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        return f"[Erro gerando roteiro: {e}]"

# --------------------------
# UI and flow
# --------------------------
st.set_page_config(page_title="Agente Roteiros Virais - Robust", layout="wide")
st.title("🎬 Agente de Roteiros Virais — Robust")

# Sidebar / controls
st.sidebar.header("Config")
api_key = st.sidebar.text_input("OpenAI API key (coloque para usar Whisper API)", type="password")
mode = st.sidebar.radio("Modo de transcrição", ["Hybrid (API preferred)", "API only", "Local only"])
whisper_model = st.sidebar.selectbox("Whisper local model (if used)", ["tiny","base","small","medium","large"], index=1)
show_debug = st.sidebar.checkbox("Mostrar debug/logs", value=True)

if api_key:
    openai.api_key = api_key

debug_area = st.empty()
def log(msg):
    if show_debug:
        debug_area.text(msg)

# Inputs
st.markdown("Cole um link do YouTube **ou** faça upload do vídeo (fallback).")
col1, col2 = st.columns(2)
with col1:
    youtube_url = st.text_input("YouTube URL")
with col2:
    uploaded = st.file_uploader("Ou faça upload do vídeo (mp4/mp3/wav)")

process_btn = st.button("Processar vídeo")

if process_btn:
    log("Iniciando pipeline...")
    audio_path = None
    # choose source
    if youtube_url:
        log("Tentando baixar áudio via yt-dlp...")
        audio_path = download_audio_from_youtube(youtube_url)
        if not audio_path:
            log("Falha no download via yt-dlp — aguardar upload manual fallback.")
    if not audio_path and uploaded:
        log("Salvando arquivo enviado...")
        tmpdir = tempfile.mkdtemp()
        tmpfile = os.path.join(tmpdir, uploaded.name)
        with open(tmpfile, "wb") as f:
            f.write(uploaded.read())
        audio_path = tmpfile

    if not audio_path:
        st.error("Nenhum arquivo/áudio disponível. Forneça um link do YouTube válido ou faça upload do arquivo.")
    else:
        transcription_text = None
        segments = None

        # API-only or Hybrid: try API first
        if mode in ("Hybrid (API preferred)","API only") and api_key:
            log("Tentando transcrever via OpenAI Whisper API...")
            api_result = transcribe_with_openai_api(audio_path)
            if api_result.get("success"):
                transcription_text = api_result.get("text")
                segments = api_result.get("segments")
                log("Transcrição via API OK.")
            else:
                err = api_result.get("error")
                log(f"API falhou com: {err} — detalhe: {api_result.get('detail')}")
                if mode == "API only":
                    st.warning("Modo API-only e a API falhou. Mude para Hybrid/Local or verifique sua quota.")
                # if hybrid, fallthrough to local
        # Local-only or fallback
        if not transcription_text and mode != "API only":
            log("Tentando transcrever via Whisper local...")
            local_result = transcribe_with_whisper_local(audio_path, model_name=whisper_model)
            if local_result.get("success"):
                transcription_text = local_result.get("text")
                segments = local_result.get("segments")
                log("Transcrição local OK.")
            else:
                log(f"Whisper local falhou: {local_result.get('detail')}")
                # try moviepy extraction then local again
                log("Tentando extrair áudio com MoviePy e reprocessar...")
                extracted = extract_audio_with_moviepy(audio_path)
                if extracted:
                    local_result2 = transcribe_with_whisper_local(extracted, model_name=whisper_model)
                    if local_result2.get("success"):
                        transcription_text = local_result2.get("text")
                        segments = local_result2.get("segments")
                        log("Transcrição local (após MoviePy) OK.")
                    else:
                        log(f"Falha após MoviePy: {local_result2.get('detail')}")
                else:
                    log("MoviePy não conseguiu extrair áudio.")

        if not transcription_text:
            st.error("Não foi possível transcrever: API e Local falharam. Verifique quota da OpenAI ou instale ffmpeg/local dependencies.")
            st.info("Dicas: 1) Se aparecer 'insufficient_quota', escolha 'Local only' e selecione 'tiny' ou 'base' para whisper_model; 2) Em máquina local instale ffmpeg (apt / brew / choco).")
        else:
            st.success("Transcrição concluída.")
            st.subheader("📜 Transcrição (trecho)")
            st.write(transcription_text[:4000] + ("..." if len(transcription_text)>4000 else ""))

            # Offer SRT if we have segments
            if segments:
                srt_text = segments_to_srt(segments)
                st.download_button("📥 Baixar legendas (.srt)", srt_text, "transcricao.srt", "text/plain")
            else:
                st.info("Sem segmentos com timestamps (SRT não disponível).")

            # Generate viral script
            log("Gerando roteiro viral com GPT (pode consumir tokens)...")
            script = generate_viral_script(transcription_text)
            st.subheader("🎯 Roteiro Viral")
            st.write(script)
            st.download_button("📥 Baixar roteiro (.txt)", script, "roteiro_viral.txt", "text/plain")

            log("Concluído.")
