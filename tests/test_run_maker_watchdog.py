"""Guards the scheduler's launcher scripts in ops/ (run-maker.ps1 watchdog, docker-heal.ps1 self-heal,
restart-docker.ps1 weekly restart). They live outside this repo (deployment, in two copies), so this runs on the
machine that has them and skips elsewhere (e.g. inside the test container). It checks the key behaviours are
present and that the two copies of each file haven't drifted."""

import hashlib
from pathlib import Path

import pytest

_DIRS = [
    Path.home() / "purffle-testrun" / "ops",
    Path.home() / "OneDrive" / "שולחן העבודה" / "shorts-scout" / "ops",
]
_FILES = ["run-maker.ps1", "docker-heal.ps1", "restart-docker.ps1", "register-docker-restart-task.ps1"]


def _copies(name):
    return [d / name for d in _DIRS if (d / name).exists()]


def _in_sync(name):
    copies = _copies(name)
    if len(copies) == 2:
        digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in copies}
        assert len(digests) == 1, f"the two {name} copies have drifted apart"


@pytest.mark.skipif(not _copies("run-maker.ps1"), reason="run-maker.ps1 ops copies not on this machine")
def test_watchdog_present_and_copies_in_sync():
    for p in _copies("run-maker.ps1"):
        text = p.read_text(encoding="utf-8")
        assert "MaxRunMinutes = 25" in text                 # 25-minute hard ceiling
        assert "WaitForExit($MaxRunMinutes" in text          # bounded wait on the container
        assert "docker kill" in text                         # kill the container on overrun
        assert "TIMEOUT" in text                             # recorded in the daily summary
        assert "retrying once" in text                       # and retried once
        assert "SKIPPED vision unavailable" in text          # fail-closed vision skip surfaced in the summary
        assert "$null = $p.Handle" in text                   # else ExitCode reads empty and every run "fails"
        assert "if ($r.Uploaded)" in text                    # a run that uploaded is never retried


@pytest.mark.skipif(not _copies("docker-heal.ps1"), reason="docker-heal.ps1 ops copies not on this machine")
def test_docker_self_heal_present_in_launcher_and_weekly_restart():
    heal = [p.read_text(encoding="utf-8") for p in _copies("docker-heal.ps1")]
    for text in heal:
        assert "--shutdown" in text                          # wsl --shutdown
        assert "desktop" in text and "stop" in text          # graceful stop first
        assert "Move-StaleSocketDirs" in text                # stale AF_UNIX sockets (error 1920) moved aside
        assert "foreach ($attempt in 1, 2)" in text          # the repair is retried once
        assert "ToastNotificationManager" in text            # Windows toast on failure
    for p in _copies("run-maker.ps1"):
        text = p.read_text(encoding="utf-8")
        assert 'docker-heal.ps1' in text and "Ensure-Docker" in text
        assert "FAILED docker" in text and "Send-Toast" in text
    for p in _copies("restart-docker.ps1"):
        text = p.read_text(encoding="utf-8")
        assert "Repair-Docker" in text and "weekly restart SKIPPED" in text   # never during a Maker run
    for name in _FILES:
        _in_sync(name)
