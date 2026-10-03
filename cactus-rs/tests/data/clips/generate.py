#!/usr/bin/env python3
"""Regenerates the synthesised speech clips the Whistle tests use.

Four short commands are spoken by Piper text-to-speech voices and written as 16 kHz mono 16-bit
WAV files, the format Whistle takes; a fifth file is one second of digital silence. `jfk.wav`
is a real recording and is not produced here (see `../README.md`).

Every setting that affects the audio is fixed below: the voice ids, the per-clip speaking rate
(`length_scale`), and Piper's two noise scales, which are zero so synthesis is deterministic.
With the same voice files and the same package versions the output is byte for byte the
committed clips; `README.md` records their sha256 digests.

Processing, per clip:

1. Piper synthesises float audio at the voice's own rate (22.05 kHz for these voices).
2. soxr resamples it to 16 kHz with its HQ preset.
3. Leading and trailing silence (|x| below 1% of the peak) is trimmed, and 100 ms of zeros is
   put back on each side.
4. The peak is normalised to -3 dBFS and the result is written as PCM_16 mono.

Usage, from anywhere:

    pip install piper-tts soundfile soxr numpy
    python cactus-rs/tests/data/clips/generate.py              # all clips, in place
    python cactus-rs/tests/data/clips/generate.py --only names_en --out /tmp/clips

Voices are downloaded from Hugging Face (`rhasspy/piper-voices`) into `--voices` on first use,
which defaults to `~/.cache/piper-voices`. Written with piper-tts 1.8.0.
"""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import download_voice

HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE = 16_000
PAD = SAMPLE_RATE // 10  # 100 ms
PEAK = 10 ** (-3 / 20)  # -3 dBFS
TRIM_THRESHOLD = 0.01  # of the peak

# name -> (Piper voice id, text, length_scale). The voices are public domain (LibriVox
# recordings) or CC0. The speaking rates were chosen so each clip runs 2 to 4 seconds and
# Whistle transcribes the intended words; `lights_de` ("Küche" heard as "Kitche") and `names_en`
# (the names without keyword biasing) are the documented exceptions the tests rely on.
CLIPS = {
    "lights_en": ("en_US-kristin-medium", "turn off the kitchen lights", 2.0),
    "lights_de": ("de_DE-thorsten_emotional-medium", "mach das Licht in der Küche aus", 1.9),
    "hall_on_en": ("en_US-joe-medium", "turn the hall light on", 2.1),
    "names_en": ("en_US-norman-medium", "please call Siobhan and Krzysztof", 2.0),
}


def synthesise(voices, voice_id, text, length_scale):
    """Speaks `text` with one voice, returning float samples and their rate."""
    model = voices / f"{voice_id}.onnx"
    if not model.exists():
        download_voice(voice_id, voices)
    voice = PiperVoice.load(str(model))
    config = SynthesisConfig(
        noise_scale=0.0,
        noise_w_scale=0.0,
        length_scale=length_scale,
        normalize_audio=False,
    )
    chunks = [chunk.audio_float_array for chunk in voice.synthesize(text, config)]
    return np.concatenate(chunks).astype(np.float32), voice.config.sample_rate


def process(samples, rate):
    """Resamples to 16 kHz, trims silence to 100 ms pads, and normalises the peak."""
    if rate != SAMPLE_RATE:
        samples = soxr.resample(samples, rate, SAMPLE_RATE, quality="HQ")
    loud = np.nonzero(np.abs(samples) >= TRIM_THRESHOLD * np.abs(samples).max())[0]
    samples = samples[loud[0] : loud[-1] + 1]
    pad = np.zeros(PAD, np.float32)
    samples = np.concatenate([pad, samples, pad])
    samples = samples / np.abs(samples).max() * PEAK
    return samples.astype(np.float32)


def write(path, samples):
    """Writes PCM_16 mono and prints the line README.md records."""
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")
    with open(path, "rb") as file:
        digest = hashlib.sha256(file.read()).hexdigest()
    print(f"{os.path.basename(path)}  {len(samples) / SAMPLE_RATE:.2f} s  {digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--voices",
        default=os.path.join(os.path.expanduser("~"), ".cache", "piper-voices"),
        help="where Piper voices are kept (downloaded on first use)",
    )
    parser.add_argument("--out", default=HERE, help="where the clips are written")
    parser.add_argument("--only", nargs="*", help="clip names to regenerate, e.g. names_en")
    args = parser.parse_args()

    voices = Path(args.voices)
    voices.mkdir(parents=True, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    for name, (voice_id, text, length_scale) in CLIPS.items():
        if args.only and name not in args.only:
            continue
        samples, rate = synthesise(voices, voice_id, text, length_scale)
        write(os.path.join(args.out, f"{name}.wav"), process(samples, rate))

    if not args.only or "silence" in args.only:
        write(os.path.join(args.out, "silence.wav"), np.zeros(SAMPLE_RATE, np.float32))


if __name__ == "__main__":
    main()
