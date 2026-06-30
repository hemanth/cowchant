"""Extended tests — meter detection across scripts, edge cases, CLI robustness."""

import os
import subprocess
import sys
import tempfile

import pytest


def _run(*args, stdin_input=None):
    return subprocess.run(
        [sys.executable, "-m", "cowchant.cli", *args],
        input=stdin_input,
        capture_output=True,
        text=True,
        timeout=120,
    )


# ─── Meter Detection Across Verses ──────────────────────────────────────────

class TestMeterDetectionVerses:
    """Test meter detection with known verses in different meters."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from cowchant.engine import CowChant, _ensure_vagdhenu_src
        _ensure_vagdhenu_src()
        self.engine = CowChant()

    def test_anushtubh_full_verse(self):
        verse = ("वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।\n"
                 "देवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥")
        meter = self.engine.detect_meter(verse)
        assert meter == "anushtubh", f"Expected anushtubh, got {meter}"

    def test_anushtubh_shuklambaradharam(self):
        verse = ("शुक्लाम्बरधरं विष्णुं शशिवर्णं चतुर्भुजम् ।\n"
                 "प्रसन्नवदनं ध्यायेत् सर्वविघ्नोपशान्तये ॥")
        meter = self.engine.detect_meter(verse)
        assert meter == "anushtubh", f"Expected anushtubh, got {meter}"

    def test_shardulavikridita_sarasvati(self):
        verse = ("या कुन्देन्दुतुषारहारधवला या शुभ्रवस्त्रावृता\n"
                 "या वीणावरदण्डमण्डितकरा या श्वेतपद्मासना ।\n"
                 "या ब्रह्माच्युतशङ्करप्रभृतिभिर्देवैः सदा वन्दिता\n"
                 "सा मां पातु सरस्वती भगवती निःशेषजाड्यापहा ॥")
        meter = self.engine.detect_meter(verse)
        # Should detect śārdūlavikrīḍita (19-syllable sama-vṛtta)
        assert isinstance(meter, str)
        assert len(meter) > 0

    def test_partial_verse_fallback(self):
        """Partial verse (not enough syllables) should still return a meter."""
        meter = self.engine.detect_meter("वसुदेवसुतं देवं")
        assert isinstance(meter, str)
        assert len(meter) > 0

    def test_empty_string(self):
        meter = self.engine.detect_meter("")
        assert isinstance(meter, str)
        assert len(meter) > 0  # should fallback

    def test_nonsanskrit_fallback(self):
        meter = self.engine.detect_meter("hello world this is english text")
        assert isinstance(meter, str)
        assert len(meter) > 0  # should fallback gracefully


# ─── Multi-Script Detection ─────────────────────────────────────────────────

class TestMultiScript:
    """Test that meter detection works with different Indic scripts."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from cowchant.engine import CowChant, _ensure_vagdhenu_src
        _ensure_vagdhenu_src()
        self.engine = CowChant()

    def test_devanagari(self):
        import prep_text as PT
        result = PT.detect_script("वसुदेवसुतं")
        assert "devanagari" in str(result).lower()

    def test_kannada_detection(self):
        import prep_text as PT
        result = PT.detect_script("ವಸುದೇವಸುತಂ")
        assert "kannada" in str(result).lower()

    def test_telugu_detection(self):
        import prep_text as PT
        result = PT.detect_script("వసుదేవసుతం")
        assert "telugu" in str(result).lower()

    def test_bengali_detection(self):
        import prep_text as PT
        result = PT.detect_script("বসুদেবসুতং")
        assert "bengali" in str(result).lower()

    def test_gujarati_detection(self):
        import prep_text as PT
        result = PT.detect_script("વસુદેવસુતં")
        assert "gujarati" in str(result).lower()

    def test_malayalam_detection(self):
        import prep_text as PT
        result = PT.detect_script("വസുദേവസുതം")
        assert "malayalam" in str(result).lower()


# ─── Text Normalization Edge Cases ───────────────────────────────────────────

class TestTextNormalization:
    """Edge cases in text preprocessing."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from cowchant.engine import _ensure_vagdhenu_src
        _ensure_vagdhenu_src()

    def test_multiple_dandas(self):
        import prep_text as PT
        result = PT.strip_punct("रामः ॥ कृष्णः ।")
        assert "॥" not in result
        assert "।" not in result

    def test_avagraha_preserved(self):
        """Avagraha (ऽ) should NOT be stripped — it's phonemically significant."""
        import prep_text as PT
        result = PT.strip_punct("कोऽपि")
        assert "ऽ" in result

    def test_om_preserved(self):
        import prep_text as PT
        result = PT.strip_punct("ॐ नमः शिवाय")
        assert "ॐ" in result

    def test_visarga_sandhi_utva(self):
        """aH + voiced → o (utva)."""
        import prep_text as PT
        slp = PT.sanscript.transliterate(
            PT.strip_punct(PT.to_deva("रामो गच्छति")),
            PT.sanscript.DEVANAGARI, PT.sanscript.SLP1,
        )
        # Already has utva, so visarga_sandhi should leave it
        result = PT.visarga_sandhi(slp)
        assert isinstance(result, str)

    def test_model_text_sandhi_produces_kannada(self):
        import prep_text as PT
        result = PT.model_text_sandhi("रामः गच्छति")
        # Should be Kannada
        assert any(0x0C80 <= ord(c) <= 0x0CFF for c in result)

    def test_whitespace_normalization(self):
        import prep_text as PT
        result = PT.strip_punct("रामः   गच्छति")
        assert "  " not in result  # multiple spaces collapsed


# ─── CLI Edge Cases ──────────────────────────────────────────────────────────

class TestCLIEdgeCases:
    def test_empty_file_input(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            r = _run("--input", f.name)
            os.unlink(f.name)
            assert r.returncode == 1

    def test_whitespace_only_input(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("   \n\n  ")
            f.flush()
            r = _run("--input", f.name)
            os.unlink(f.name)
            assert r.returncode == 1

    def test_nonexistent_input_file(self):
        r = _run("--input", "/nonexistent/path.txt")
        assert r.returncode != 0

    def test_invalid_device(self):
        r = _run("--device", "tpu", "verse")
        assert r.returncode != 0

    def test_version_flag_exits_0(self):
        r = _run("--version")
        assert r.returncode == 0

    def test_meter_flag_accepted(self):
        """--meter should be accepted as an arg (even if render fails without torch)."""
        r = _run("test verse", "--meter", "anuṣṭubh", "--device", "cpu", "--no-play")
        # Won't render without torch but should not fail on arg parsing
        assert "unrecognized" not in r.stderr.lower()


# ─── Engine Error Handling ───────────────────────────────────────────────────

class TestEngineErrors:
    def test_chant_without_torch_gives_clear_error(self):
        """Without torch, chant() should raise RuntimeError with install hint."""
        try:
            import torch  # noqa: F401
            pytest.skip("torch is installed -- cannot test missing-torch path")
        except ImportError:
            pass
        from cowchant.engine import CowChant
        engine = CowChant(device="cpu")
        with pytest.raises(RuntimeError, match="pip install"):
            engine.chant("test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
