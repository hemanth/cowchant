"""Terminal spinner with Sanskrit fun facts -- shown during model load and inference."""

import sys
import threading
import time
import random
import os

# Fun facts shown while waiting
FUN_FACTS = [
    "Vagdhenu means 'the wish-cow of speech' (vak + dhenu)",
    "Sanskrit has 48 phonemes -- every one is precisely defined by Panini",
    "The Anustubh meter (8+8+8+8 syllables) accounts for ~75%% of all Sanskrit verse",
    "Vasantatilaka ('spring's forehead mark') has 14 syllables per line",
    "Sardula-vikridita ('tiger's sport') is the longest common meter at 19 syllables",
    "Sanskrit sandhi rules merge words at boundaries -- there are 18 vowel sandhi types",
    "The DiT backbone uses 337M parameters with 22 transformer layers",
    "BigVGAN converts mel spectrograms to waveforms at 24kHz sample rate",
    "Vagdhenu achieves MOS ~4.6 from expert Vedic listeners",
    "The visarga (H) transforms before voiced consonants -- a key sandhi rule",
    "Sragdhara ('garland-bearer') has 21 syllables per line -- the longest standard meter",
    "All Sanskrit consonant clusters are perfectly handled, including retroflex aspirates",
    "The reference bank contains per-meter audio clips for prosody control",
    "Flow-matching (OT-CFM) generates mel spectrograms in 64 denoising steps",
    "Panini's Ashtadhyayi (~400 BCE) is the world's first formal grammar",
    "The anusvara becomes homorganic before stops: n before t, ng before k",
    "Cowchant supports all major Indic scripts -- Devanagari, Kannada, Telugu, and more",
    "The chant pipeline: text -> SLP1 -> Kannada -> DiT -> BigVGAN -> audio",
]

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Spinner:
    """A terminal spinner that shows fun facts while work happens in the background."""

    def __init__(self, message="Loading"):
        self._message = message
        self._stop = threading.Event()
        self._thread = None
        self._facts = list(FUN_FACTS)
        random.shuffle(self._facts)
        self._fact_idx = 0

    def _spin(self):
        i = 0
        last_fact_time = 0
        fact_text = ""
        while not self._stop.is_set():
            frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
            now = time.monotonic()
            if now - last_fact_time > 4.0 or not fact_text:
                fact_text = self._facts[self._fact_idx % len(self._facts)]
                self._fact_idx += 1
                last_fact_time = now

            line = f"\r{frame} {self._message}  --  {fact_text}"
            # Truncate to terminal width
            try:
                cols = os.get_terminal_size().columns
                if len(line) > cols - 1:
                    line = line[: cols - 4] + "..."
            except OSError:
                pass

            sys.stderr.write(f"\r\033[2K{line}")
            sys.stderr.flush()
            i += 1
            self._stop.wait(0.1)

        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def update(self, message):
        self._message = message

    def stop(self, final_message=None):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if final_message:
            sys.stderr.write(f"\r\033[2K{final_message}\n")
            sys.stderr.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()
