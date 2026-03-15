"""Tests for src/config.py"""

from pathlib import Path
from src.config import get_config, get_project_root, resolve_path


def test_get_config_returns_dict():
    cfg = get_config()
    assert isinstance(cfg, dict)


def test_config_has_required_sections():
    cfg = get_config()
    for section in ["paths", "data_filters", "features", "training", "mlflow", "api"]:
        assert section in cfg, f"Missing section: {section}"


def test_config_paths_are_strings():
    cfg = get_config()
    for key, val in cfg["paths"].items():
        assert isinstance(val, str), f"Path '{key}' should be a string, got {type(val)}"


def test_get_project_root_exists():
    root = get_project_root()
    assert root.exists()
    assert (root / "configs" / "config.yaml").exists()


def test_resolve_path():
    resolved = resolve_path("configs/config.yaml")
    assert resolved.is_absolute()
    assert resolved.exists()


def test_phase_definitions_complete():
    cfg = get_config()
    phases = cfg["features"]["phase_definitions"]
    for phase_name in ["powerplay", "middle", "death"]:
        assert phase_name in phases
        assert "from_over" in phases[phase_name]
        assert "to_over" in phases[phase_name]
