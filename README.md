# OmniVoice

1-click [Pinokio](https://pinokio.computer) launcher for [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) — a zero-shot, massively multilingual text-to-speech model covering **600+ languages**.

- **Voice cloning** — clone any voice from a 3–10 second reference clip (Whisper auto-transcribes it if you leave the transcript blank).
- **Voice design** — describe a voice with attributes (`male, british accent, low pitch`) and get it without any reference audio.
- **Auto voice** — let the model pick a voice for you.
- **Inline controls** — non-verbal tags (`[laughter]`, `[sigh]`, …), pinyin tone overrides for Chinese, and CMU phoneme overrides for English.
- Runs on NVIDIA CUDA, Apple Silicon (MPS) and CPU. Apache-2.0 licensed.

The launcher clones the upstream repository into `app/`, installs it into a local virtual environment, and pre-downloads the model weights so the first launch is fast.

## How to use

1. Click **Install**. This installs the Python environment and downloads:
   - `k2-fsa/OmniVoice` (~3.3 GB) — the TTS model
   - `openai/whisper-large-v3-turbo` (~1.6 GB) — used only to auto-transcribe reference audio
2. Click **Start**. The web UI opens automatically once the model is loaded.
3. In the UI, pick a tab:
   - **Voice Clone** — upload/record a short reference clip, type your text, generate.
   - **Voice Design** — describe the voice instead of uploading audio.
4. Generated audio can be played in the browser or downloaded as a `.wav` file.

**Update** pulls the latest upstream source, reinstalls the package and refreshes the model weights.
**Reset** removes `app/` and `env/` so the next Install starts from scratch. Models stay in the Hugging Face cache.

### Tips

- Reference clips of **3–10 seconds** work best; longer clips slow cloning down and can hurt quality.
- For the cleanest pronunciation, use a reference clip in the **same language** as the target text.
- Normalize numbers to words first (`123` → `one hundred twenty-three`) when using the Python/CLI APIs.
- The UI reads a fixed starting port from Pinokio (`{{port}}`), so it never collides with another app.

## API

The web UI is a Gradio app, so it also exposes a programmatic API on its own address (`http://127.0.0.1:<port>` — copy it from the **Open Web UI** link).

### Python

Install the client package once: `uv pip install omnivoice` (or reuse the launcher's `env`).

```python
import soundfile as sf
import torch
from omnivoice import OmniVoice

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",   # "mps" on Apple Silicon, "xpu" on Intel Arc
    dtype=torch.float16,
)

# Voice cloning (omit ref_text to auto-transcribe with Whisper)
audio = model.generate(
    text="Hello, this is a test of zero-shot voice cloning.",
    ref_audio="ref.wav",
    ref_text="Transcription of the reference audio.",
)

# Voice design (no reference audio)
audio = model.generate(
    text="Hello, this is zero-shot voice design.",
    instruct="female, low pitch, british accent",
)

sf.write("out.wav", audio[0], 24000)   # list of np.ndarray at 24 kHz
```

Reuse a cloned voice across sessions:

```python
prompt = model.create_voice_clone_prompt(ref_audio="ref.wav", ref_text="...")
prompt.save("my_voice.pt")

from omnivoice import VoiceClonePrompt
prompt = VoiceClonePrompt.load("my_voice.pt")
audio = model.generate(text="Hello again!", voice_clone_prompt=prompt)
```

Generation knobs: `num_step=32` (`16` for faster), `speed=1.0`, `duration=10.0` (seconds, overrides speed).

### Command line

```bash
omnivoice-infer --model k2-fsa/OmniVoice \
  --text "This is a test for text to speech." \
  --ref_audio ref.wav --ref_text "Transcription of the reference audio." \
  --output hello.wav
```

```bash
omnivoice-infer --model k2-fsa/OmniVoice \
  --text "This is a test for text to speech." \
  --instruct "male, British accent" \
  --output hello.wav
```

```bash
omnivoice-demo --ip 127.0.0.1 --port 8001        # the web UI
omnivoice-infer-batch --model k2-fsa/OmniVoice --test_list test.jsonl --res_dir results/
```

### HTTP (Gradio API)

The demo is served by Gradio, so its endpoint index is available as JSON:

```bash
curl http://127.0.0.1:7860/gradio_api/info
```

Use it from Python with `gradio_client`:

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860")
client.view_api()   # lists the available endpoints
```

…or from JavaScript with `@gradio/client`:

```javascript
import { Client } from "@gradio/client";
const client = await Client.connect("http://127.0.0.1:7860");
const api = await client.view_api();
console.log(api);
```

## Layout

```
omniVoice/
├── app/            # cloned upstream repository (created by Install)
│   └── env/        # python virtual environment (created by Install)
├── icon.png
├── install.js      # clone + install + download weights
├── start.js        # launch the Gradio web UI on the next free port
├── update.js       # git pull + reinstall + refresh weights
├── reset.js        # remove app/ and env/
├── pinokio.js      # dynamic UI
└── pinokio.json    # metadata
```

## Notes

- On Windows/Linux with an NVIDIA GPU the install pulls the CUDA 12.8 build of PyTorch automatically (via the project's `pyproject.toml` index config); on Apple Silicon it gets the PyPI build with MPS support.
- AMD GPUs on Linux fall back to the ROCm build via `torch.js`; on other AMD setups the app runs on CPU.
- Upstream project: <https://github.com/k2-fsa/OmniVoice>
