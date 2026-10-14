# 🎨 Voice → Image Studio

Speak in **Tamil or English**, get an accurate **English text**, and turn it into an **AI-generated image**, all from a simple Streamlit web app.

```
🎤 Voice  →  📝 Speech-to-text (Whisper)  →  🌐 Tamil→English (NLLB)  →  ✏️ Edit prompt  →  🖼️ Image (Stable Diffusion)
```

<!-- Add a screenshot of the app here:
![App screenshot](docs/screenshot.png)
-->

## ✨ Features

- 🎙️ **In-browser voice recording** and 📁 **WAV/MP3 upload**
- 🗣️ **Tamil and English** speech recognition with a manual language selector (or auto-detect)
- 🌐 **Tamil → English translation** using Meta's NLLB model
- ✏️ **Editable English prompt**, so you can fix a mistake before generating
- 🎨 **Style presets** (Realistic, Digital art, Cartoon, Anime, Watercolor, Oil painting)
- 🎛️ Controls for quality steps, prompt strength, negative prompt and **seed**
- 🕘 History of generated images and PNG download
- ⚡ Runs on **CPU** or **GPU**, and switches automatically

## 🧰 Tech stack

| Task | Tool |
|---|---|
| Speech recognition | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`small` / `medium`) |
| Translation | [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) |
| Image generation | [segmind/tiny-sd](https://huggingface.co/segmind/tiny-sd) via 🤗 Diffusers |
| UI | Streamlit |

## 📁 Project structure

```
.
├── app.py                          # Streamlit app (the whole pipeline)
├── requirements.txt                # Python dependencies
├── Speech_To_Image_Converter.ipynb # One-click Google Colab launcher
├── .gitignore
└── README.md
```

---

## 🚀 Option 1: Run on Google Colab (recommended, free GPU)

Fastest option: a full request takes about 15 seconds on a T4 GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Speech_To_Image_Converter.ipynb)

1. Click the **Open in Colab** badge above.
2. Choose **Runtime → Change runtime type → T4 GPU**, then save.
3. Run the cells **in order, top to bottom**:
   1. **Check GPU**: should print `True Tesla T4`.
   2. **Clone the repo** and install dependencies.
   3. **Install CUDA 12 libraries** needed by faster-whisper.
   4. **Download cloudflared**, the tunnel that gives you a public link.
   5. **Start the app**: prints a `https://....trycloudflare.com` link.
4. Open the link and use the app. Allow microphone access when the browser asks.
5. When finished, run the last **Stop** cell to shut down the app and tunnel.

> **Notes**
> - Keep the "start the app" cell running while you use the app.
> - The link changes every time you start the tunnel.
> - Colab sessions are temporary. After a disconnect, re-run the cells. Models re-download on a fresh session.

### Manual Colab setup (without the notebook)

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install -q -r requirements.txt nvidia-cublas-cu12 nvidia-cudnn-cu12
```
Then follow the cloudflared and start steps in the notebook.

---

## 💻 Option 2: Run locally (Windows / macOS / Linux)

Works on CPU only (8 GB RAM minimum). Expect roughly 1-2 minutes per image.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

python -m venv venv
# Windows (PowerShell):  .\venv\Scripts\Activate.ps1
# macOS / Linux:         source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the models (about 3-4 GB in total). After that they are cached.

---

## 🎯 How to use

1. In the **sidebar**, pick the spoken language. Choosing it manually is more accurate than auto-detect.
2. **Record** your voice or **upload** a WAV/MP3 file.
3. Click **Transcribe & Translate**.
4. Check the English text and **edit it if needed**.
5. Pick a **style** and click **Generate Image**.
6. Download the PNG, or change the style or seed and generate again.

## ⚙️ Settings guide

| Setting | What it does |
|---|---|
| Speech model | `medium` is much better for Tamil; `small` is faster |
| Style | Adds style keywords after your sentence |
| Quality steps | More steps = more detail, slower |
| Prompt strength | Higher = follows your text more strictly |
| Negative prompt | Things the image should avoid |
| Seed | Same seed + same prompt = same image |

## 🛠️ Troubleshooting

| Error | Fix |
|---|---|
| `open() got an unexpected keyword argument 'metadata_errors'` | PyAV version mismatch. Already avoided: the app decodes audio with `librosa`. Make sure you have the latest `app.py`. |
| `Library libcublas.so.12 is not found` | Install `nvidia-cublas-cu12 nvidia-cudnn-cu12` and start Streamlit with `LD_LIBRARY_PATH` set (the notebook does this). |
| Link doesn't open / app shows old behaviour | An old process is still running. Run the **Stop** cell, then the **Start** cell, and open the **new** link. |
| Mic button doesn't record | Use the `https://` link and allow microphone access. |
| GPU out of memory | Switch the speech model to `small`. |
| Very slow on local machine | Normal on CPU. Use Colab, or lower the quality steps. |

## ⚠️ Limitations

- Speech recognition and translation are not perfect, especially for noisy audio, strong accents or rare words. That is why the English prompt is editable.
- Tiny-SD is a small, fast model, so image detail is lower than larger Stable Diffusion models.
- Currently supports **pure Tamil** or **pure English** (no mixed "Tanglish").

## 🔮 Future ideas

- Mixed Tamil-English (Tanglish) support
- More languages through NLLB
- Tamil fine-tuned Whisper model
- Optional larger image models (SD 1.5 / SDXL-Turbo) for GPU users
- Image-to-image and style reference upload

## 🙏 Acknowledgements

[OpenAI Whisper](https://github.com/openai/whisper) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [Meta NLLB](https://ai.meta.com/research/no-language-left-behind/) · [Segmind Tiny-SD](https://huggingface.co/segmind/tiny-sd) · [Hugging Face Diffusers](https://github.com/huggingface/diffusers) · [Streamlit](https://streamlit.io)

## 📄 License

MIT. See `LICENSE` (add one via GitHub: *Add file → Create new file → LICENSE → choose template*).