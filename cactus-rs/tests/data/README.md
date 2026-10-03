# tests/data

`smart_home.json` is the smart-home acceptance suite from Cactus Compute's `needle` repository
(`needle/environments/smart_home.py` at commit `94df999`), transcribed to JSON with its system
prompt, five tool declarations and thirty-two frozen query/expected-call cases intact.

It is used here under the Apache-2.0 licence of its source, which the file records in its own
`source` and `license` fields; those fields are part of the fixture and are kept verbatim.

## clips/

Sixteen-bit mono WAV files at 16 kHz, the format Whistle takes, used by `tests/whistle_tests.rs`
and the `transcribe` and `voice_lights` examples. Each is read as `i16 / 32768.0`.

### jfk.wav

- **Spoken text:** "And so, my fellow Americans, ask not what your country can do for you, ask
  what you can do for your country."
- **Source:** `tests/jfk.flac` in the `openai/whisper` repository (MIT licensed repository). The
  recording is from President John F. Kennedy's inaugural address of 20 January 1961, a work of
  the US federal government and so in the public domain.
- **Processing:** 44.1 kHz stereo FLAC, downmixed to mono and resampled to 16 kHz, written as
  PCM_16: 176 000 samples, 11.0 s.
- **sha256:** `a5d96a7e096feff0727b9a86d38ebac21099b61071fb2ca08d59fa2d221beeb6`

### The synthesised clips

The other five files were made for this repository by `clips/generate.py`, which documents its
own settings and reproduces them byte for byte (`pip install piper-tts soundfile soxr numpy`,
then `python cactus-rs/tests/data/clips/generate.py`). The four spoken ones use voices from
`rhasspy/piper-voices` on Hugging Face, synthesised with piper-tts 1.8.0, both noise scales at
zero so the output is deterministic, and this processing: Piper's 22.05 kHz output resampled to
16 kHz with soxr's HQ preset, leading and trailing silence trimmed with 100 ms pads kept, the
peak normalised to -3 dBFS, written as PCM_16 mono.

| File | Spoken text | Piper voice | Voice licence | length_scale | Samples | sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| `lights_en.wav` | "turn off the kitchen lights" | `en_US-kristin-medium` | public domain (LibriVox) | 2.0 | 46 697 | `a3c1234c45b17e76faf077d1fea3041a31daf5cbeb0370ad2a082f50831a8805` |
| `lights_de.wav` | "mach das Licht in der Küche aus" | `de_DE-thorsten_emotional-medium` | CC0 | 1.9 | 51 118 | `258769c758701b4b7b8857dba2bc0c658bb26f4a6c2731bd0474b271d563386c` |
| `hall_on_en.wav` | "turn the hall light on" | `en_US-joe-medium` | CC0 | 2.1 | 32 893 | `6107ac60b2054f7f838275a68f49facbce68283f055527dc666e3bb9eca6262b` |
| `names_en.wav` | "please call Siobhan and Krzysztof" | `en_US-norman-medium` | public domain (LibriVox) | 2.0 | 47 081 | `ac04315348819ba60673d6a6efa81fa9c82e1c351ff670438f8adeacabf9119d` |
| `silence.wav` | none: one second of zeros | none | n/a | n/a | 16 000 | `643f8a8dc8bd9c19225afffad2becfec5426180b3749cb208abdf1a6c8354efc` |

What Whistle hears in them, measured with the pinned engine and with upstream's `cactus-needle`
3.1.0 alike:

- `lights_en.wav`: "Turn off the kitchen lights." (English)
- `lights_de.wav`: "Mach das Licht in der Kitche aus." (German; "Küche" comes out as "Kitche",
  so the tests check only for "licht" and "aus")
- `hall_on_en.wav`: "Turn the hall light on." (English)
- `names_en.wav`: "Please call Chevorne and Crystal." with no keywords, and "Please call Shivorn
  and Krzysztof." with the keywords `Siobhan` and `Krzysztof`
- `silence.wav`: an empty transcript with no language
