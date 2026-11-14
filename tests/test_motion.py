"""Less like a slideshow: 2.5D parallax from a depth map, a cut every 3-5 s with motion transitions, and a
quiet ambient bed of the place's sound under the voice."""

import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from purffle_shorts import ambience as A
from purffle_shorts import depth as D
from purffle_shorts import ffmpeg as FF
from purffle_shorts import render as R
from purffle_shorts.config import Settings
from purffle_shorts.media import MediaItem
from purffle_shorts.timing import SceneTiming, Word, split_shots


# ------------------------------------------------------------------ 1. depth maps
@pytest.mark.parametrize("w, h", [(1600, 1067), (1080, 1920), (4000, 3000), (600, 600)])
def test_the_model_input_keeps_the_aspect_in_multiples_of_14(w, h):
    iw, ih = D.input_size(w, h)
    assert iw % 14 == 0 and ih % 14 == 0
    assert max(iw, ih) <= D.MAX_LONG_SIDE + 14
    assert abs(iw / ih - w / h) < 0.08
    if max(w, h) / min(w, h) < 1.7:
        assert abs(min(iw, ih) - D.INPUT_SIDE) <= 14


def test_a_depth_map_is_normalised_smoothed_and_near_is_white():
    import numpy as np
    raw = np.tile(np.linspace(0, 10, 100, dtype=np.float32)[:, None], (1, 60))[None]   # nearer towards the bottom
    img = D.to_map(raw, (30, 50))
    a = np.asarray(img)
    assert img.size == (30, 50) and img.mode == "L"
    assert a[-3:].mean() > 200 and a[:3].mean() < 55


def test_no_model_means_no_maps(monkeypatch, tmp_path):
    monkeypatch.setenv("DEPTH_DIR", str(tmp_path / "none"))
    assert D.maps_for([tmp_path / "a.jpg"], tmp_path) == {}


def test_maps_come_from_a_child_process(monkeypatch, tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "model.onnx").write_bytes(b"x")
    monkeypatch.setenv("DEPTH_DIR", str(tmp_path / "m"))
    seen = {}

    def run(cmd, input, **k):
        seen["cmd"] = cmd
        pairs = json.loads(input)
        Path(pairs[0][1]).write_bytes(b"png")           # the second image failed in the child
        return SimpleNamespace(returncode=0, stderr="depth failed for b.jpg: bad image")
    monkeypatch.setattr(D.subprocess, "run", run)
    out = D.maps_for([tmp_path / "a.jpg", tmp_path / "b.jpg"], tmp_path / "w")
    assert seen["cmd"][1:] == ["-m", "purffle_shorts.depth"]
    assert list(out) == [tmp_path / "a.jpg"]


def test_a_crashed_child_leaves_the_plain_zoom(monkeypatch, tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    monkeypatch.setenv("DEPTH_DIR", str(tmp_path))

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("python", 300)
    monkeypatch.setattr(D.subprocess, "run", boom)
    assert D.maps_for([tmp_path / "a.jpg"], tmp_path / "w") == {}


# ------------------------------------------------------------------ 2. the parallax segment
def test_the_parallax_graph_moves_near_pixels_more_than_far_ones():
    g = R._parallax_graph(1080, 1920, 30, 135, 4.5, "", random.Random(1))
    assert "displace=edge=smear" in g and "zoompan=" in g and ":d=1:" in g and "s=1080x1920" in g
    assert "geq=lum='128+lum(X,Y)/255*" in g and "(2*T/4.500-1)" in g      # far (0) stays, near moves, over time
    assert g.endswith("[v]") and "format=gbrp" in g                          # every plane displaced alike


def _image(tmp_path, depth=True):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"x")
    item = MediaItem("wikimedia", "1", "image", path=img, width=1600, height=1067)
    if depth:
        (tmp_path / "d.png").write_bytes(b"x")
        item.extra["depth"] = str(tmp_path / "d.png")
    return item


def test_a_picture_with_a_depth_map_gets_parallax(monkeypatch, tmp_path):
    labels = []
    monkeypatch.setattr(FF, "run", lambda args, label="", **k: labels.append(label))
    R.prepare_segment(_image(tmp_path), 4.0, 0, Settings(), tmp_path)
    assert labels == ["segment 0 (parallax)"]


def test_parallax_failing_falls_back_to_the_zoom(monkeypatch, tmp_path):
    labels = []

    def run(args, label="", **k):
        labels.append(label)
        if "parallax" in label:
            raise FF.FFmpegError("displace failed")
    monkeypatch.setattr(FF, "run", run)
    R.prepare_segment(_image(tmp_path), 4.0, 0, Settings(), tmp_path)
    assert labels == ["segment 0 (parallax)", "segment 0 (ken burns)"]


def test_no_map_or_parallax_off_keeps_the_zoom(monkeypatch, tmp_path):
    labels = []
    monkeypatch.setattr(FF, "run", lambda args, label="", **k: labels.append(label))
    R.prepare_segment(_image(tmp_path, depth=False), 4.0, 0, Settings(), tmp_path)
    R.prepare_segment(_image(tmp_path), 4.0, 1, Settings(parallax=False), tmp_path)
    assert labels == ["segment 0 (ken burns)", "segment 1 (ken burns)"]


# ------------------------------------------------------------------ 3. pacing: a cut every 3-5 s, motion transitions
def _words(seconds):
    return [Word(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(int(seconds / 0.4))]


@pytest.mark.parametrize("seconds, shots", [(7.0, 2), (8.0, 2), (5.5, 1), (4.0, 1), (10.0, 2)])
def test_shots_are_three_to_five_seconds(seconds, shots):
    s = Settings()
    out = split_shots([SceneTiming(0, 0.0, seconds)], _words(seconds), s.max_shot_seconds,
                      min_len=s.min_shot_seconds)
    assert len(out) == shots
    if shots > 1:
        assert all(3.0 <= st.duration <= 5.0 + 0.5 for st in out)


def test_motion_transitions_are_whip_pans_and_push_ins_never_twice_in_a_row():
    seq = R.motion_transitions(200, random.Random(3))
    assert set(seq) <= set(R.WHIP_PANS + R.PUSH_INS) and "zoomin" in seq
    assert all(a != b for a, b in zip(seq, seq[1:]))
    assert 0.15 < seq.count("zoomin") / len(seq) < 0.45


def test_defaults():
    s = Settings()
    assert s.transition == "motion" and s.max_shot_seconds == 5.0 and s.min_shot_seconds == 3.0
    assert s.parallax and s.ambience and 0 < s.transition_seconds <= 0.35


# ------------------------------------------------------------------ 4. the ambient bed
@pytest.mark.parametrize("text, bed", [
    ("The Mogao Caves are a system of 500 Buddhist cave temples near Dunhuang", "temple"),
    ("Tabo Monastery is a gompa in Spiti", "temple"),
    ("Bako National Park on the coast of Sarawak, with sea stacks and beaches", "sea"),
    ("MacRitchie Reservoir is Singapore's oldest reservoir", "stream"),
    ("Horton Plains is a plateau of grassland and cloud forest", "forest"),
    ("Konye-Urgench is the ruined capital of Khwarezm in the desert", "wind"),
])
def test_the_bed_matches_the_place(text, bed):
    assert A.choose(text) == bed


def test_a_missing_recording_falls_back_to_synthesised_wind(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBIENCE_DIR", str(tmp_path))
    b = A.bed("temple", 50.0)
    assert b.name == "wind" and "anoisesrc=color=brown" in " ".join(b.inputs) and b.lufs == A.WIND_LUFS
    (tmp_path / "temple.ogg").write_bytes(b"x")
    b = A.bed("temple", 50.0)
    assert b.name == "temple" and b.inputs[:2] == ["-stream_loop", "-1"] and b.lufs == A.FILES["temple"][1]


def test_the_bed_is_mixed_under_the_voice_and_ducked(monkeypatch, tmp_path):
    graphs = []

    def run(args, **k):
        graphs.append(args[args.index("-filter_complex") + 1] if "-filter_complex" in args else "")
        return ""
    monkeypatch.setattr(FF, "run", run)
    monkeypatch.setattr(R, "_measure", lambda path, filters: -16.0)
    monkeypatch.setattr(R, "_voice_filters", lambda s: "")
    monkeypatch.setattr(FF, "loudnorm_report", lambda err: {})
    s = Settings(ambience_below_voice=24.0)
    bed = A.Bed("temple", ["-stream_loop", "-1", "-i", str(tmp_path / "t.ogg")], "", -29.5)
    R.mix_audio(s, tmp_path / "voice.wav", 50.0, None, tmp_path, ambience=bed)
    g = graphs[0]
    gain = 10 ** ((-16.0 - 24.0 + 29.5) / 20)
    assert f"volume={gain:.5f}" in g and "[amb]" in g
    assert "[v0]asplit=2[voice][sc0]" in g and "[amb][sc0]sidechaincompress" in g
    assert "amix=inputs=2" in g


def test_music_and_bed_both_duck(monkeypatch, tmp_path):
    graphs = []
    monkeypatch.setattr(FF, "run", lambda args, **k: graphs.append(args[args.index("-filter_complex") + 1]) or "")
    monkeypatch.setattr(R, "_measure", lambda path, filters: -16.0)
    monkeypatch.setattr(R, "_voice_filters", lambda s: "")
    monkeypatch.setattr(FF, "loudnorm_report", lambda err: {})
    bed = A.bed("wind", 50.0)
    R.mix_audio(Settings(), tmp_path / "v.wav", 50.0, tmp_path / "m.mp3", tmp_path, ambience=bed)
    g = graphs[0]
    assert "asplit=3[voice][sc0][sc1]" in g and "[mus][sc0]" in g and "[amb][sc1]" in g and "lowpass=f=700" in g
