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

## images/

### stop.png

A 512 x 384 drawing made for this repository: a red octagon with the word STOP in white on a
grey pole, over a green band and a light blue background. Used by `tests/mirage_tests.rs`, the
`cactus-sys` probe `tests/probe_vision.rs` and the `describe` example. It is not a photograph and
has no third-party source. ImageMagick 7.1.1 drew it:

```sh
magick -size 512x384 xc:'#8fb4d9' \
  -fill '#5c7a3a' -draw 'rectangle 0,260 512,384' \
  -fill '#777777' -draw 'rectangle 250,150 262,330' \
  -fill '#c8102e' -draw 'polygon 226,60 286,60 328,102 328,162 286,204 226,204 184,162 184,102' \
  -fill white -font Helvetica-Bold -pointsize 46 -gravity NorthWest -annotate +199+113 'STOP' \
  -depth 8 -strip -define png:exclude-chunks=date,time stop.png
```

The lettering depends on the Helvetica installed where the command runs, so another machine
gives slightly different pixels; the committed file is the fixture.

- **sha256:** `8fa4f810c2486152f3b4cd01dae09f94391fa660d47ce00362d960319981eb52`
- **Geometry, in pixels:** the octagon spans x 184 to 328 and y 60 to 204; the pole x 250 to 262
  and y 150 to 330; the green band starts at y 260.

What Mirage sees in it, measured with the pinned engine on macOS arm64:

- objects: a "stop sign" at `[361, 174, 642, 517]` on the 0 to 1000 scale, a "pole" and "grass"
- text: one line, "STOP", at `[370, 283, 632, 407]`
- relations: the sign `attached_to` the pole
- a form asking `stop`, `yield` or `speed limit` answers `stop` with probability 0.77
