import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent


def _candidate_paths(*relative_paths: str) -> list[Path]:
    candidates: list[Path] = []
    for rel in relative_paths:
        candidates.append(REPO_ROOT / rel)
        candidates.append(APP_DIR / rel)
    return candidates


def resolve_model_path(
    env_var: str,
    relative_candidates: list[str],
    display_name: str,
) -> Path:
    env_value = os.getenv(env_var)
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.exists():
            return env_path

    for path in _candidate_paths(*relative_candidates):
        if path.exists():
            return path

    searched = [str(p) for p in _candidate_paths(*relative_candidates)]
    if env_value:
        searched.insert(0, env_value)

    raise FileNotFoundError(
        f"{display_name} not found. Set {env_var} or place it in one of: {searched}"
    )


def get_flood_model_path() -> Path:
    return resolve_model_path(
        env_var="FLOOD_MODEL_PATH",
        relative_candidates=[
            "models/flood_model_final.pth",
            "flood_model_final.pth",
            "depth_classifier.pth",
        ],
        display_name="Flood model weights",
    )


def get_severity_model_path() -> Path:
    return resolve_model_path(
        env_var="SEVERITY_MODEL_PATH",
        relative_candidates=[
            "models/severity_efficientnet.pth",
            "models/severity_model.pth",
            "severity_efficientnet.pth",
            "severity_model.pth",
        ],
        display_name="Severity model weights",
    )
