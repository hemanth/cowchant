"""CowChant engine — wraps Vāgdhenu's render_core.Renderer with weight management.

Handles:
- Auto-downloading model weights from HuggingFace
- Device detection (CUDA → MPS → CPU)
- Lazy model loading (only on first .chant() call)
- BigVGAN clone-on-demand
- WAV output via soundfile
"""

import json
import os
import sys
import subprocess

import numpy as np

# HuggingFace model repo for weights
WEIGHTS_REPO = "prathoshap/vagdhenu"
VOICE_FILE = "voice_steer_ema_2026-06-17.pt"
VOC_FILE = "voc_bigvgan_EMA_2026-06-11.pth"
VOCAB_FILE = "vocab.txt"

# Default cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "cowchant")


def _detect_device():
    """Auto-detect best available device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _ensure_bigvgan():
    """Verify BigVGAN is importable (installed as pip dependency)."""
    try:
        import bigvgan  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "BigVGAN is not installed.\n"
            "  Reinstall cowchant: pip install cowchant"
        )


def _ensure_indicf5():
    """Ensure IndicF5 is importable -- clone if not installed."""
    try:
        import f5_tts  # noqa: F401
        return
    except ImportError:
        pass
    dst = os.path.join(CACHE_DIR, "IndicF5")
    if not os.path.isdir(os.path.join(dst, ".git")):
        os.makedirs(CACHE_DIR, exist_ok=True)
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/ai4bharat/IndicF5.git", dst,
            ],
            check=True, capture_output=True,
        )
    if dst not in sys.path:
        sys.path.insert(0, dst)


def _ensure_vagdhenu_src():
    """Clone vagdhenu src for render_core + text frontend."""
    dst = os.path.join(CACHE_DIR, "vagdhenu")
    if not os.path.isdir(os.path.join(dst, ".git")):
        os.makedirs(CACHE_DIR, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/prathoshap/vagdhenu.git",
                dst,
            ],
            check=True,
            capture_output=True,
        )
    src = os.path.join(dst, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


class CowChant:
    """Sanskrit chant TTS engine.

    Usage::

        engine = CowChant()
        engine.chant("वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।", output="output.wav")

        # Or get raw audio
        sr, audio = engine.chant("शुक्लाम्बरधरं विष्णुं...")
    """

    def __init__(self, device=None, speed=0.90, nfe=64, cfg=3.0):
        self.device = device or _detect_device()
        self.speed = speed
        self.nfe = nfe
        self.cfg = cfg
        self._renderer = None
        self._src_dir = None

    def _lazy_init(self):
        """Download weights + load models on first use."""
        if self._renderer is not None:
            return

        import warnings
        from cowchant.spinner import Spinner

        # 0. Check torch is installed
        try:
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "cowchant requires PyTorch for inference.\n"
                "  Install with: pip install cowchant\n"
                "  Or manually: pip install torch torchaudio"
            )

        spinner = Spinner("Downloading model weights")
        spinner.start()

        try:
            from huggingface_hub import hf_hub_download

            # 1. Ensure dependencies are importable
            _ensure_bigvgan()
            _ensure_indicf5()

            # 1b. Patch BigVGAN for huggingface_hub >= 1.x compatibility
            import bigvgan as _bv
            _orig_fp = _bv.BigVGAN._from_pretrained.__func__

            @classmethod
            def _patched_fp(cls, *args, proxies=None, resume_download=None, **kwargs):
                return _orig_fp(cls, *args, proxies=proxies, resume_download=resume_download, **kwargs)

            _bv.BigVGAN._from_pretrained = _patched_fp

            self._src_dir = _ensure_vagdhenu_src()

            # 2. Download weights from HuggingFace
            voice = hf_hub_download(WEIGHTS_REPO, VOICE_FILE)
            voc = hf_hub_download(WEIGHTS_REPO, VOC_FILE)

            # 3. Resolve vocab.txt
            vocab = os.path.join(self._src_dir, "reference_bank", VOCAB_FILE)
            if not os.path.exists(vocab):
                try:
                    vocab = hf_hub_download(WEIGHTS_REPO, VOCAB_FILE)
                except Exception:
                    vocab = None

            # 4. Bank path
            bank = os.path.join(self._src_dir, "reference_bank", "bank.json")

            # 5. Load renderer (suppress verbose upstream output)
            spinner.update(f"Loading models on {self.device}")
            from render_core import Renderer

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Redirect stdout to suppress jieba/vocos chatter
                _real_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
                try:
                    self._renderer = Renderer(
                        voice, voc, bank, device=self.device, vocab_file=vocab,
                        speed=self.speed, nfe=self.nfe, cfg=self.cfg,
                    )
                finally:
                    sys.stdout.close()
                    sys.stdout = _real_stdout

        finally:
            spinner.stop("cowchant: ready!")

    def meters(self):
        """Return list of supported meter names."""
        src = _ensure_vagdhenu_src()
        bank_path = os.path.join(src, "reference_bank", "bank.json")
        bank = json.load(open(bank_path, encoding="utf-8"))
        return [
            k
            for k, v in bank.items()
            if not k.startswith("_") and isinstance(v, dict) and "wav" in v
        ]

    def detect_meter(self, text):
        """Auto-detect the meter (chandas) of a verse. Pure text -- no GPU needed."""
        _ensure_vagdhenu_src()
        try:
            import re
            from indic_transliteration import sanscript
            import prep_text as PT
            from tts_syllabify import syllabify
            from tts_weight import tag_weights
            from tts_meter import detect_meter as _detect

            d = PT.to_deva(text).replace("\u0965", "|").replace("\u0964", "|").replace("\n", " | ")
            d = "".join(c for c in d if not (c.isdigit() or ("\u0966" <= c <= "\u096f")) and c not in "\"'\u201c\u201d\u2018\u2019()")
            slp = re.sub(r"\s+", " ", sanscript.transliterate(d, sanscript.DEVANAGARI, sanscript.SLP1)).strip()
            syls = syllabify(slp)
            tag_weights(syls)
            name = _detect(syls).get("name", "unknown")
            if name in ("anushtubh_half", "anushtubh"):
                return "anushtubh"
            if name in ("unknown", None, ""):
                return "vasantatilak\u0101"
            return name
        except Exception:
            return "vasantatilak\u0101"

    def chant(self, text, output=None, meter=None, seed=60):
        """Synthesize a Sanskrit verse into chanted audio.

        Args:
            text: Sanskrit verse in any Indic script (Devanagari, Kannada, Telugu, etc.)
            output: Optional output WAV file path. If None, returns (sr, audio).
            meter: Override auto-detected meter. Use .meters() for options.
            seed: Random seed for variation (default: 60).

        Returns:
            If output is None: tuple (sample_rate, audio_numpy_float32)
            If output is set: the output file path
        """
        import warnings
        from cowchant.spinner import Spinner

        self._lazy_init()

        # Auto-detect meter if not specified
        if meter is None:
            meter = self.detect_meter(text)

        spinner = Spinner(f"Chanting in {meter}")
        spinner.start()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sr, audio = self._renderer.render_one(text, meter, seed=seed)
        finally:
            spinner.stop()

        if output:
            import soundfile as sf

            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
            sf.write(output, audio, sr)
            sys.stderr.write(f"Saved: {output} ({len(audio)/sr:.1f}s, {meter})\n")
            return output

        return sr, audio

