"""A quiet ambient bed matching the place - temple chimes, water, sea, forest or wind - mixed under the voice
and ducked whenever it speaks (render.mix_audio).

The recordings are CC0 / public domain, from Wikimedia Commons; the Dockerfile downloads them and checks their
SHA-256 (a file that changed upstream fails the build):

* temple  "Windglockenspiel.Koshi.ogg" - Koshi wind chimes, by Membeth (CC0)
* stream  "433589 jackthemurray stream-river-water-up-close.wav" - a soft stream, by jackthemurray (CC0)
* sea     "Ocean Waves on a Tropical Beach.ogg" - by Jarrod Stanley / J.D. Savanyu (CC0)
* forest  "20090610 0 ambience.ogg" - forest ambience, by nille (public domain)

Wind (deserts, mountains, ruins - and the default) is synthesised from brown noise with slow gusts, like the
sound effects in sfx.py: no file and no licence. A bed whose file is missing falls back to wind.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

AMBIENCE_DIR = "/opt/ambience"

# name: (file, integrated loudness in LUFS as measured on the pinned file)
FILES = {
    "temple": ("temple.ogg", -29.5),
    "stream": ("stream.wav", -53.9),
    "sea": ("sea.ogg", -11.2),
    "forest": ("forest.ogg", -23.0),
}
WIND_LUFS = -26.5         # measured on the synthesised wind (60 s, seed 7)
_WIND_SRC = "anoisesrc=color=brown:amplitude=0.6:sample_rate=48000:duration={d:.2f}:seed={seed}"
_WIND_FX = ("lowpass=f=700,highpass=f=90,"
            "volume='0.55+0.45*sin(2*PI*0.071*t)*sin(2*PI*0.031*t+1.3)':eval=frame")

# First match wins: a temple by the sea gets the temple's chimes.
RULES = [
    ("temple", r"temples?|monaster(?:y|ies)|gompas?|pagodas?|stupas?|shrines?|wats?|dzongs?|grottoe?s?|"
               r"buddhist|hindu|viharas?|chortens?|lamaser(?:y|ies)|cave temples?"),
    ("sea", r"beach(?:es)?|coast(?:al|line)?|islands?|atolls?|lagoons?|reefs?|bays?|seas?|ocean|seaside|"
            r"harbou?rs?|water village|stilt village"),
    ("stream", r"rivers?|streams?|waterfalls?|lakes?|reservoirs?|wetlands?|mangroves?|springs?|canals?|creeks?|"
               r"brooks?"),
    ("forest", r"forests?|jungles?|rainforests?|woodlands?|national park|gardens?|groves?"),
]


@dataclass
class Bed:
    name: str
    inputs: list[str]       # ffmpeg input arguments
    filters: str            # filters on that input before the mix gain
    lufs: float


def directory() -> Path:
    return Path(os.getenv("AMBIENCE_DIR") or AMBIENCE_DIR)


def choose(*texts: str) -> str:
    """The bed for a place from its name, its article's opening and the narration."""
    blob = " ".join(t for t in texts if t).lower()
    for name, pattern in RULES:
        if re.search(rf"\b(?:{pattern})\b", blob):
            return name
    return "wind"


def bed(name: str, total: float, seed: int = 7) -> Bed:
    """ffmpeg inputs for the bed ``name`` (looped to cover ``total`` seconds); wind when its file is missing."""
    if name in FILES:
        f, lufs = FILES[name]
        path = directory() / f
        if path.exists():
            return Bed(name, ["-stream_loop", "-1", "-i", str(path.resolve())], "", lufs)
    src = _WIND_SRC.format(d=total + 1.0, seed=seed)
    return Bed("wind", ["-f", "lavfi", "-i", src], _WIND_FX, WIND_LUFS)


def for_place(total: float, *texts: str) -> Bed:
    return bed(choose(*texts), total)
