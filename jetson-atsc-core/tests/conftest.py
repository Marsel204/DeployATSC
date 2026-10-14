"""Isolated local fixtures; tests never write into an operator's profiles."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def local_audit_only(monkeypatch):
    """Unit/fixture validation never discovers an operator's external API keys."""
    from src.audit.llm_auditor import LLMAuditor
    monkeypatch.setattr(LLMAuditor, "_discover_api_key", lambda self: None)


@pytest.fixture
def isolated_config_output(tmp_path, monkeypatch):
    import src.ui.session_manager as sessions
    (tmp_path / "configs").mkdir()
    monkeypatch.setattr(sessions, "CORE_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def video_factory(tmp_path):
    def make(name="test.mp4", size=(320, 240), fps=25, seconds=2, color=(40, 40, 40)):
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        assert writer.isOpened()
        for index in range(int(fps * seconds)):
            frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            cv2.putText(frame, str(index), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            writer.write(frame)
        writer.release()
        return path
    return make
