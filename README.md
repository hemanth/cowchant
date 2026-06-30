# 🐄 cowchant

Sanskrit chant TTS on the command line — powered by [Vāgdhenu](https://github.com/prathoshap/vagdhenu).

Turn Sanskrit verses into traditional metered chant. Paste a śloka, get audio.

> **MOS ~4.6** (expert listener). Handles all Sanskrit conjuncts including retroflex aspirates (ṣṭ, ḍḍh, …) with 100% accuracy.

## Install

```bash
pip install cowchant

# With GPU support (recommended — 10x faster):
pip install cowchant[gpu]

# Full dependencies (includes transformers, accelerate):
pip install cowchant[full]
```

> **Note:** First run downloads ~2GB of model weights from HuggingFace + clones IndicF5 and BigVGAN. Subsequent runs use the cached models.

## Usage

```bash
# Basic — auto-detects meter, outputs to output.wav
cowchant "वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।"

# Specify output file and meter
cowchant "शुक्लाम्बरधरं विष्णुं..." -o vishnu.wav --meter anuṣṭubh

# From file
cowchant --input verse.txt -o chant.wav

# From stdin
echo "गुरुर्ब्रह्मा गुरुर्विष्णुः..." | cowchant -o guru.wav

# List supported meters
cowchant --list-meters

# Force CPU (slower but works without GPU)
cowchant "verse..." --device cpu

# Change seed for a different take
cowchant "verse..." --seed 42
```

## Supported scripts

Works with **any Indian script** — Devanagari, Kannada, Telugu, Malayalam, Bengali, Gujarati, Gurmukhi, Oriya, Grantha. Auto-detected.

## Python API

```python
from cowchant.engine import CowChant

engine = CowChant(device="cuda")

# Save to file
engine.chant("वसुदेवसुतं देवं...", output="chant.wav")

# Get raw audio
sr, audio = engine.chant("शुक्लाम्बरधरं विष्णुं...")

# List meters
print(engine.meters())
```

## Options

| Flag | Default | Description |
|---|---|---|
| `-o, --output` | `output.wav` | Output WAV path |
| `-m, --meter` | auto | Override meter (chandas) |
| `-s, --seed` | `60` | Random seed for variation |
| `-i, --input` | — | Read verse from file |
| `--device` | auto | `cuda` / `mps` / `cpu` |
| `--speed` | `0.90` | Chant speed |
| `--nfe` | `64` | DiT denoising steps |
| `--cfg` | `3.0` | CFG strength |
| `--list-meters` | — | List supported meters |

## How it works

- **Backbone:** IndicF5 / F5-TTS (flow-matching DiT, ~337M params)
- **Vocoder:** NVIDIA BigVGAN-v2, fine-tuned
- **Text frontend:** Devanagari → SLP1 → Kannada routing, visarga sandhi, homorganic anusvāra, meter/gaṇa detection
- **Reference bank:** Per-meter reference audio clips for prosody control

## Performance

| Device | Time per śloka |
|---|---|
| NVIDIA GPU (A100/4090) | ~5 seconds |
| Apple MPS (M1/M2) | ~20 seconds |
| CPU | ~60+ seconds |

## Credits

- **Engine:** [Vāgdhenu](https://github.com/prathoshap/vagdhenu) by Prof. Prathosh, IISc Bengaluru
- **CLI:** [Hemanth HM](https://h3manth.com)
- **License:** Apache-2.0

## Etymology

**Vāgdhenu** = vāk (speech) + dhenu (cow) — "the wish-cow of speech."
**cowchant** = cowsay vibes + the *dhenu* from the project name. 🐄
