import tempfile

from market_jepa import tempfiles


def test_lab_temp_is_the_default_when_no_override(monkeypatch, tmp_path):
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.delenv("MARKET_JEPA_TMPDIR", raising=False)
    monkeypatch.setattr(tempfiles, "LAB_TMP_ROOT", tmp_path)
    previous = tempfile.tempdir
    try:
        destination = tempfiles.configure_tempdir()
        assert destination.parent == tmp_path
        assert destination.is_dir()
        assert tempfile.gettempdir() == str(destination)
    finally:
        tempfile.tempdir = previous


def test_explicit_temp_override_wins(monkeypatch, tmp_path):
    destination = tmp_path / "job-specific"
    monkeypatch.setenv("MARKET_JEPA_TMPDIR", str(destination))
    previous = tempfile.tempdir
    try:
        assert tempfiles.configure_tempdir() == destination
        assert destination.is_dir()
    finally:
        tempfile.tempdir = previous
