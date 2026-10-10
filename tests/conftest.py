"""Never make real Gemini requests in the regression suite."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


@pytest.fixture(autouse=True)
def offline_gemini_configuration(tmp_path, monkeypatch, request):
    # Subprocess CLI parity must also run explicitly with Gemini disabled.
    settings = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    settings = deepcopy(settings)
    settings["inference"]["llm_judge"].update(enabled=False, google_api_key="")
    settings["inference"]["video"] = {"gemini_review_enabled": False}
    # Unit tests use explicit mocked collectors; only the approved checkpoint
    # smoke exercises installed diagnostic models and caches.
    real_evidence = request.node.name == "test_approved_image_and_saved_video_frame_real_parity"
    settings["inference"]["v6_evidence"] = {**settings["inference"].get("v6_evidence", {}), "enabled": real_evidence}
    path = tmp_path / "offline-config.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setenv("FLOOD_CONFIG_PATH", str(path))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from src import settings as loader
    original = loader._load_yaml
    def offline_load(path):
        data = original(path)
        if "llm_judge" in data.get("inference", {}):
            data["inference"]["llm_judge"].update(enabled=False, google_api_key="")
            data["inference"]["v6_evidence"] = {**data["inference"].get("v6_evidence", {}), "enabled": real_evidence}
        return data
    monkeypatch.setattr(loader, "_load_yaml", offline_load)
    from src.llm_judge import LLMJudge
    def no_network(*args, **kwargs):
        raise AssertionError("Real Gemini request prohibited in tests")
    monkeypatch.setattr(LLMJudge, "_call_google_api", no_network)
