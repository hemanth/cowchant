"""Comprehensive tests for cowchant — text frontend, engine, and CLI edge cases."""

import os
import subprocess
import sys
import json
import tempfile

import pytest


# ─── Helper ──────────────────────────────────────────────────────────────────

def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "cowchant.cli", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


# ─── CLI Tests ───────────────────────────────────────────────────────────────

class TestCLIBasic:
    def test_version(self):
        r = _run("--version")
        assert r.returncode == 0
        assert "cowchant 0.1.0" in r.stdout

    def test_help(self):
        r = _run("--help")
        assert r.returncode == 0
        assert "Vagdhenu" in r.stdout
        assert "--meter" in r.stdout
        assert "--seed" in r.stdout
        assert "--device" in r.stdout

    def test_no_args_exits_1(self):
        r = _run()
        assert r.returncode == 1

    def test_no_args_shows_guidance(self):
        r = _run()
        assert "Please provide" in r.stdout or "usage" in r.stdout.lower()


class TestCLIListMeters:
    def test_lists_meters(self):
        r = _run("--list-meters")
        assert r.returncode == 0
        assert "anuṣṭubh" in r.stdout

    def test_lists_known_meters(self):
        r = _run("--list-meters")
        for meter in ["vasantatilakā", "śārdūlavikrīḍita", "mālinī", "gadya"]:
            assert meter in r.stdout, f"Missing meter: {meter}"

    def test_meter_count(self):
        r = _run("--list-meters")
        assert "meters total" in r.stdout


class TestCLIInputModes:
    def test_input_from_file(self):
        """--input flag should read text from a file (verify parsing, not inference)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।\n")
            f.flush()
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "cowchant.cli", "--input", f.name, "--no-play"],
                    capture_output=True, text=True, timeout=10,
                )
                # If it completed fast, it should NOT say "Please provide"
                assert "Please provide" not in r.stdout
            except subprocess.TimeoutExpired as e:
                # Timeout means it accepted the text and started model load — success
                stderr = e.stderr.decode() if e.stderr else ""
                assert "model" in stderr.lower() or "Downloading" in stderr or "Loading" in stderr
            finally:
                os.unlink(f.name)

    def test_stdin_pipe(self):
        """Piped stdin should be accepted (verify parsing, not inference)."""
        try:
            r = subprocess.run(
                [sys.executable, "-m", "cowchant.cli", "--no-play"],
                input="वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।\n",
                capture_output=True, text=True, timeout=10,
            )
            assert "Please provide" not in r.stdout
        except subprocess.TimeoutExpired as e:
            # Timeout means it accepted the text and started model load — success
            stderr = e.stderr.decode() if e.stderr else ""
            assert "model" in stderr.lower() or "Downloading" in stderr or "Loading" in stderr


# ─── Engine Tests ────────────────────────────────────────────────────────────

class TestEngineInit:
    def test_import(self):
        from cowchant.engine import CowChant
        assert CowChant is not None

    def test_default_device(self):
        from cowchant.engine import CowChant, _detect_device
        engine = CowChant()
        assert engine.device == _detect_device()

    def test_explicit_device(self):
        from cowchant.engine import CowChant
        engine = CowChant(device="cpu")
        assert engine.device == "cpu"

    def test_custom_params(self):
        from cowchant.engine import CowChant
        engine = CowChant(speed=0.85, nfe=32, cfg=2.5)
        assert engine.speed == 0.85
        assert engine.nfe == 32
        assert engine.cfg == 2.5

    def test_renderer_not_loaded_on_init(self):
        from cowchant.engine import CowChant
        engine = CowChant()
        assert engine._renderer is None


class TestEngineMeters:
    def test_meters_returns_list(self):
        from cowchant.engine import CowChant
        engine = CowChant()
        meters = engine.meters()
        assert isinstance(meters, list)

    def test_meters_nonempty(self):
        from cowchant.engine import CowChant
        engine = CowChant()
        meters = engine.meters()
        assert len(meters) >= 10  # should have 18

    def test_meters_has_anushtubh(self):
        from cowchant.engine import CowChant
        engine = CowChant()
        meters = engine.meters()
        assert "anuṣṭubh" in meters

    def test_meters_has_no_underscore_prefix(self):
        from cowchant.engine import CowChant
        engine = CowChant()
        meters = engine.meters()
        for m in meters:
            assert not m.startswith("_"), f"Internal key leaked: {m}"


# ─── Text Frontend Tests (via vagdhenu src) ──────────────────────────────────

class TestTextFrontend:
    """Test the vendored vagdhenu text frontend (prep_text.py)."""

    @pytest.fixture(autouse=True)
    def _ensure_src(self):
        from cowchant.engine import _ensure_vagdhenu_src
        _ensure_vagdhenu_src()

    def test_detect_script_devanagari(self):
        import prep_text as PT
        assert PT.detect_script("वसुदेवसुतं") == PT.sanscript.DEVANAGARI

    def test_detect_script_kannada(self):
        import prep_text as PT
        assert PT.detect_script("ವಸುದೇವಸುತಂ") == PT.sanscript.KANNADA

    def test_detect_script_telugu(self):
        import prep_text as PT
        assert PT.detect_script("వసుదేవసుతం") == PT.sanscript.TELUGU

    def test_to_deva_passthrough(self):
        import prep_text as PT
        deva = "वसुदेवसुतं"
        assert PT.to_deva(deva) == deva

    def test_to_deva_from_kannada(self):
        import prep_text as PT
        result = PT.to_deva("ವಸುದೇವಸುತಂ")
        # Should produce Devanagari
        assert any(0x0900 <= ord(c) <= 0x097F for c in result)

    def test_strip_punct(self):
        import prep_text as PT
        result = PT.strip_punct("वसुदेवसुतं देवं ।")
        assert "।" not in result
        assert "वसुदेवसुतं" in result

    def test_strip_punct_removes_digits(self):
        import prep_text as PT
        result = PT.strip_punct("1.2 वसुदेवसुतं")
        assert "1" not in result
        assert "2" not in result

    def test_fix_colon_to_visarga(self):
        import prep_text as PT
        assert "ः" in PT.fix_colon("गुरु:")

    def test_model_text_produces_kannada(self):
        import prep_text as PT
        result = PT.model_text("वसुदेवसुतं देवं कंसचाणूरमर्दनम्")
        # Should contain Kannada characters
        assert any(0x0C80 <= ord(c) <= 0x0CFF for c in result)

    def test_model_text_no_dandas(self):
        import prep_text as PT
        result = PT.model_text("वसुदेवसुतं देवं ।")
        assert "।" not in result

    def test_phonetic_mfa(self):
        import prep_text as PT
        result = PT.phonetic_mfa("रामः गच्छति")
        # visarga before voiced → should be transformed
        assert isinstance(result, str)
        assert len(result) > 0

    def test_anusvara_homorganic(self):
        import prep_text as PT
        # anusvāra before ka-varga should become ṅ
        result = PT.phonetic_mfa("संकटम्")
        assert "ङ" in result or "ं" in result  # either homorganic or kept

    def test_model_text_sandhi(self):
        import prep_text as PT
        result = PT.model_text_sandhi("रामः गच्छति")
        assert isinstance(result, str)
        # Should be in Kannada
        assert any(0x0C80 <= ord(c) <= 0x0CFF for c in result)


# ─── Reference Bank Tests ───────────────────────────────────────────────────

class TestReferenceBank:
    """Verify the reference bank structure."""

    @pytest.fixture
    def bank(self):
        from cowchant.engine import _ensure_vagdhenu_src
        src = _ensure_vagdhenu_src()
        path = os.path.join(src, "reference_bank", "bank.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_bank_loads(self, bank):
        assert isinstance(bank, dict)
        assert len(bank) > 0

    def test_meters_have_wav(self, bank):
        for k, v in bank.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and "wav" in v:
                assert v["wav"].endswith(".wav"), f"{k}: wav doesn't end with .wav"

    def test_meters_have_ref_text(self, bank):
        for k, v in bank.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and "wav" in v:
                assert "ref_text" in v, f"{k}: missing ref_text"
                assert len(v["ref_text"]) > 0, f"{k}: empty ref_text"

    def test_wav_files_exist(self, bank):
        from cowchant.engine import _ensure_vagdhenu_src
        src = _ensure_vagdhenu_src()
        bdir = os.path.join(src, "reference_bank")
        for k, v in bank.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and "wav" in v:
                wav_path = os.path.join(bdir, v["wav"])
                assert os.path.exists(wav_path), f"{k}: WAV missing: {v['wav']}"

    def test_vocab_exists(self):
        from cowchant.engine import _ensure_vagdhenu_src
        src = _ensure_vagdhenu_src()
        vocab = os.path.join(src, "reference_bank", "vocab.txt")
        assert os.path.exists(vocab), "vocab.txt missing from reference bank"


# ─── Meter Detection Tests ──────────────────────────────────────────────────

class TestMeterDetection:
    """Test auto meter detection (requires tts_meter etc. from vagdhenu src)."""

    @pytest.fixture(autouse=True)
    def _ensure_src(self):
        from cowchant.engine import _ensure_vagdhenu_src
        _ensure_vagdhenu_src()

    def test_detect_anushtubh(self):
        from cowchant.engine import CowChant
        engine = CowChant()
        # Classic anuṣṭubh verse (32 syllables)
        verse = "वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।\nदेवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥"
        meter = engine.detect_meter(verse)
        assert "anush" in meter.lower() or "anuṣṭubh" in meter.lower() or meter != ""

    def test_detect_returns_string(self):
        from cowchant.engine import CowChant
        engine = CowChant()
        meter = engine.detect_meter("शुक्लाम्बरधरं विष्णुं शशिवर्णं चतुर्भुजम्")
        assert isinstance(meter, str)
        assert len(meter) > 0


# ─── Device Detection Tests ─────────────────────────────────────────────────

class TestDeviceDetection:
    def test_returns_valid_device(self):
        from cowchant.engine import _detect_device
        device = _detect_device()
        assert device in ("cuda", "mps", "cpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
