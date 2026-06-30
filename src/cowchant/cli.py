#!/usr/bin/env python3
"""cowchant -- Sanskrit chant TTS on the command line.

Usage:
    cowchant "वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।"
    cowchant "शुक्लाम्बरधरं विष्णुं..." -o vishnu.wav --meter anuṣṭubh
    cowchant --list-meters
    cowchant --input verses.txt -o chants/output.wav
    echo "गुरुर्ब्रह्मा..." | cowchant -o guru.wav
"""

import argparse
import os
import platform
import subprocess
import sys

from cowchant import __version__


def _build_parser():
    p = argparse.ArgumentParser(
        prog="cowchant",
        description="cowchant -- Sanskrit chant TTS, powered by Vagdhenu (DiT + BigVGAN)",
        epilog="Developed by Prof. Prathosh (IISc) · CLI by Hemanth HM · Apache-2.0",
    )
    p.add_argument("text", nargs="?", default=None, help="Sanskrit verse in any Indic script")
    p.add_argument("-o", "--output", default="output.wav", help="Output WAV path (default: output.wav)")
    p.add_argument("-m", "--meter", default=None, help="Override auto-detected meter (chandas)")
    p.add_argument("-s", "--seed", type=int, default=60, help="Random seed for variation (default: 60)")
    p.add_argument("-i", "--input", dest="input_file", default=None, help="Read verse from file")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"], help="Compute device (default: auto)")
    p.add_argument("--speed", type=float, default=0.90, help="Chant speed (default: 0.90)")
    p.add_argument("--nfe", type=int, default=64, help="NFE steps for DiT (default: 64)")
    p.add_argument("--cfg", type=float, default=3.0, help="CFG strength (default: 3.0)")
    p.add_argument("--list-meters", action="store_true", help="List all supported meters and exit")
    p.add_argument("--no-play", action="store_true", help="Don't play audio after synthesis")
    p.add_argument("--version", action="version", version=f"cowchant {__version__}")
    return p


def _play_audio(path):
    """Play a WAV file using the platform's native player."""
    try:
        if platform.system() == "Darwin":
            subprocess.run(["afplay", path], check=True)
        elif platform.system() == "Linux":
            # Try aplay (ALSA), then paplay (PulseAudio)
            for cmd in ["aplay", "paplay"]:
                try:
                    subprocess.run([cmd, path], check=True)
                    return
                except FileNotFoundError:
                    continue
        else:
            # Windows or unknown — skip
            pass
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # --list-meters: lightweight, no GPU needed
    if args.list_meters:
        from cowchant.engine import CowChant

        engine = CowChant()
        meters = engine.meters()
        print("Supported meters (chandas):\n")
        for m in meters:
            print(f"  • {m}")
        print(f"\n  ({len(meters)} meters total)")
        return

    # Resolve input text
    text = args.text

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif text is None:
        # Try reading from stdin
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()

    if not text:
        parser.print_help()
        print("\nPlease provide a Sanskrit verse as an argument, via --input, or via stdin.")
        sys.exit(1)

    # Chant!
    from cowchant.engine import CowChant

    engine = CowChant(device=args.device, speed=args.speed, nfe=args.nfe, cfg=args.cfg)

    try:
        result = engine.chant(text, output=args.output, meter=args.meter, seed=args.seed)
        if not args.no_play and result and os.path.isfile(result):
            _play_audio(result)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
