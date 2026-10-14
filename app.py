import io
import os
import re
import tempfile
import time

import streamlit as st
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =========================================================
# Page config + styling
# =========================================================
st.set_page_config(page_title="Voice → Image Studio", page_icon="🎨", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #4f46e5 0%, #9333ea 55%, #ec4899 100%);
        color: white;
    }
    .hero h1 { margin: 0; font-size: 2rem; color: white; }
    .hero p  { margin: .3rem 0 0 0; opacity: .92; }
    .step-title { font-weight: 700; font-size: 1.05rem; margin-bottom: .2rem; }
    .pill {
        display: inline-block; padding: 2px 12px; border-radius: 999px;
        background: rgba(147,51,234,.15); color: #9333ea; font-size: .8rem;
        font-weight: 600; margin-right: 6px;
    }
    </style>
    <div class="hero">
        <h1>🎨 Voice → Image Studio</h1>
        <p>Speak in Tamil or English → accurate English text → AI-generated image</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Device
# =========================================================
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
DTYPE = torch.float16 if USE_GPU else torch.float32
CPU_THREADS = max(1, (os.cpu_count() or 4) - 1)

STYLES = {
    "Realistic photo": "photorealistic, highly detailed, sharp focus, natural lighting, 4k",
    "Digital art": "digital art, concept art, vibrant colors, intricate details, artstation",
    "Cartoon": "cartoon illustration, bold outlines, bright flat colors, clean background",
    "Anime": "anime style, studio quality, vivid colors, detailed, soft shading",
    "Watercolor": "watercolor painting, soft brush strokes, pastel tones, artistic",
    "Oil painting": "oil painting, rich textures, dramatic lighting, masterpiece",
    "Random": "",
}
NEGATIVE_DEFAULT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, extra limbs, "
    "watermark, text, signature, cropped"
)

# =========================================================
# Model loaders (each cached separately, loaded lazily)
# =========================================================
@st.cache_resource(show_spinner="Loading speech model...")
def load_whisper(size: str):
    compute = "float16" if USE_GPU else "int8"  # int8 keeps RAM low on CPU
    return WhisperModel(size, device=DEVICE, compute_type=compute, cpu_threads=CPU_THREADS)


@st.cache_resource(show_spinner="Loading translation model...")
def load_translator():
    model_id = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
    mdl.eval()
    return tok, mdl


@st.cache_resource(show_spinner="Loading image model...")
def load_sd():
    pipe = StableDiffusionPipeline.from_pretrained(
        "segmind/tiny-sd", torch_dtype=DTYPE, safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


# =========================================================
# Pipeline steps
# =========================================================
def load_audio(path: str):
    """Decode to 16 kHz mono float32 ourselves (avoids PyAV version issues)."""
    import librosa

    audio, _ = librosa.load(path, sr=16000, mono=True)
    return audio


def transcribe(audio_path: str, language: str, model_size: str):
    """Returns (text, detected_language, confidence)."""
    model = load_whisper(model_size)
    audio_path = load_audio(audio_path)  # numpy array instead of file path
    kwargs = dict(
        task="transcribe",  # never 'translate' - we want the real Tamil text first
        beam_size=5,
        vad_filter=True,  # strips silence -> fewer hallucinations
        condition_on_previous_text=False,
        temperature=0.0,
    )

    if language in ("ta", "en"):
        segments, info = model.transcribe(audio_path, language=language, **kwargs)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, language, 1.0

    # Auto-detect, but only choose between Tamil and English
    segments, info = model.transcribe(audio_path, language=None, **kwargs)
    text = " ".join(s.text.strip() for s in segments).strip()
    lang, prob = info.language, info.language_probability
    probs = dict(getattr(info, "all_language_probs", None) or [])
    if lang not in ("ta", "en") and probs:
        lang = "ta" if probs.get("ta", 0) >= probs.get("en", 0) else "en"
        prob = probs.get(lang, prob)
        segments, _ = model.transcribe(audio_path, language=lang, **kwargs)
        text = " ".join(s.text.strip() for s in segments).strip()
    return text, lang, prob


def translate_ta_to_en(text: str) -> str:
    tok, mdl = load_translator()
    tok.src_lang = "tam_Taml"
    # Translate sentence by sentence - NLLB is more accurate on short inputs
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]
    if not sentences:
        return ""
    inputs = tok(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.inference_mode():
        out = mdl.generate(
            **inputs,
            forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
            num_beams=5,
            max_length=128,
        )
    return " ".join(tok.batch_decode(out, skip_special_tokens=True)).strip()


def build_prompt(text: str, style: str) -> str:
    # Subject first (CLIP reads only ~77 tokens), style keywords after
    suffix = STYLES[style]
    return f"{text.strip().rstrip('.')}, {suffix}" if suffix else text.strip()


def generate_image(prompt, negative, steps, guidance, seed):
    pipe = load_sd()
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    with torch.inference_mode():
        return pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            height=512,
            width=512,
            generator=gen,
        ).images[0]


def save_temp(uploaded) -> str:
    ext = os.path.splitext(getattr(uploaded, "name", "") or "")[1].lower()
    if ext not in (".wav", ".mp3"):
        ext = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        f.write(uploaded.getbuffer())
        return f.name


# =========================================================
# Session state
# =========================================================
for key, default in {
    "original": "",
    "english": "",
    "lang": "",
    "history": [],
    "last_image": None,
    "last_meta": "",
}.items():
    st.session_state.setdefault(key, default)

# =========================================================
# Sidebar settings
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"Running on: **{'GPU 🚀' if USE_GPU else 'CPU 🐢'}**")

    st.subheader("🗣️ Speech")
    lang_choice = st.selectbox(
        "Spoken language",
        ["Tamil (தமிழ்)", "English", "Auto-detect (Tamil/English)"],
        help="Selecting the language manually is the most accurate option.",
    )
    lang_code = {"Tamil (தமிழ்)": "ta", "English": "en"}.get(lang_choice, "auto")

    model_size = st.selectbox(
        "Speech model",
        ["small", "medium"],
        index=1,
        help="'medium' is much better at Tamil but slower. Use 'small' for quick tests.",
    )

    st.subheader("🎨 Image")
    style = st.selectbox("Style", list(STYLES.keys()))
    steps = st.slider("Quality steps", 10, 40, 25 if not USE_GPU else 30,
                      help="More steps = better detail but slower.")
    guidance = st.slider("Prompt strength", 4.0, 12.0, 7.5, 0.5,
                         help="Higher = follows your text more strictly.")
    negative = st.text_area("Negative prompt (what to avoid)", NEGATIVE_DEFAULT, height=90)

    st.subheader("🎲 Seed")
    randomize = st.checkbox("Random seed each time", value=True)
    seed_val = st.number_input("Seed", 0, 2**31 - 1, 42, disabled=randomize,
                               help="Same seed + same prompt = same image.")

    if st.button("🗑️ Clear history", width="stretch"):
        st.session_state.history = []
        st.session_state.last_image = None

# =========================================================
# STEP 1 - Audio input
# =========================================================
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="step-title">① Give your voice input</div>', unsafe_allow_html=True)
    tab_rec, tab_up = st.tabs(["🎙️ Record", "📁 Upload"])

    audio_file = None
    with tab_rec:
        recorded = st.audio_input("Click the mic, speak, then stop")
        if recorded is not None:
            audio_file = recorded
    with tab_up:
        uploaded = st.file_uploader("WAV or MP3", type=["wav", "mp3"])
        if uploaded is not None:
            st.audio(uploaded)
            audio_file = uploaded

    if st.button("📝 Transcribe & Translate", type="primary", width="stretch",
                 disabled=audio_file is None):
        path = save_temp(audio_file)
        try:
            with st.status("Processing your voice...", expanded=True) as status:
                t0 = time.time()
                st.write("🎤 Recognizing speech...")
                text, lang, prob = transcribe(path, lang_code, model_size)

                if not text:
                    status.update(label="No speech detected", state="error")
                    st.warning("I couldn't hear any speech. Please try again closer to the mic.")
                else:
                    st.session_state.original = text
                    st.session_state.lang = lang
                    if lang == "ta":
                        st.write("🌐 Translating Tamil → English...")
                        st.session_state.english = translate_ta_to_en(text)
                    else:
                        st.session_state.english = text
                    status.update(label=f"Done in {time.time() - t0:.1f}s", state="complete")
        finally:
            if os.path.exists(path):
                os.remove(path)

# =========================================================
# STEP 2 - Review / edit text
# =========================================================
with right:
    st.markdown('<div class="step-title">② Review & edit the text</div>', unsafe_allow_html=True)
    if st.session_state.original:
        lang_name = "Tamil" if st.session_state.lang == "ta" else "English"
        st.markdown(f'<span class="pill">Detected: {lang_name}</span>', unsafe_allow_html=True)
        if st.session_state.lang == "ta":
            st.text_area("🗣️ What I heard (Tamil)", st.session_state.original,
                         height=80, disabled=True)
        st.session_state.english = st.text_area(
            "🌐 English prompt (edit if anything is wrong)",
            st.session_state.english,
            height=110,
        )
        final_prompt = build_prompt(st.session_state.english, style)
        with st.expander("See the final enhanced prompt"):
            st.code(final_prompt, language=None)

        if st.button("🎨 Generate Image", type="primary", width="stretch",
                     disabled=not st.session_state.english.strip()):
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if randomize else int(seed_val)
            with st.spinner("Painting your idea... (CPU may take 1-2 minutes)"):
                t0 = time.time()
                img = generate_image(final_prompt, negative, steps, guidance, seed)
                took = time.time() - t0
            st.session_state.last_image = img
            st.session_state.last_meta = f"Seed {seed} · {steps} steps · {took:.0f}s · {style}"
            st.session_state.history.insert(
                0, {"img": img, "text": st.session_state.english, "meta": st.session_state.last_meta}
            )
    else:
        st.info("Your transcript and English translation will appear here.")

# =========================================================
# STEP 3 - Result
# =========================================================
st.divider()
st.markdown('<div class="step-title">③ Your image</div>', unsafe_allow_html=True)

if st.session_state.last_image is not None:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(st.session_state.last_image, caption=st.session_state.last_meta, width="stretch")
    with c2:
        buf = io.BytesIO()
        st.session_state.last_image.save(buf, format="PNG")
        st.download_button("💾 Download PNG", buf.getvalue(), "voice_image.png",
                           "image/png", width="stretch")
        st.caption("Not happy? Edit the English text above, or change style / seed, and generate again.")
else:
    st.caption("Nothing generated yet.")

if len(st.session_state.history) > 1:
    st.markdown("#### 🕘 History")
    cols = st.columns(4)
    for i, item in enumerate(st.session_state.history[:8]):
        with cols[i % 4]:
            st.image(item["img"], width="stretch")
            st.caption(item["text"][:60])