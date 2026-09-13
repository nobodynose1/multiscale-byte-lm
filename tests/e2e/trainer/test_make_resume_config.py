import importlib.util
from pathlib import Path

import pytest

module_path = Path(__file__).with_name("make_resume_config.py")
module_spec = importlib.util.spec_from_file_location("make_resume_config", module_path)
assert module_spec is not None and module_spec.loader is not None
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
build_resume_config = module.build_resume_config


def test_build_resume_config_sets_checkpoint_and_cumulative_target() -> None:
    config = {"train": {"target_elements": 10_240}}

    result = build_resume_config(config, Path("run-1/latest.pth"), 20_480)

    assert result["train"]["target_elements"] == 20_480
    assert result["resume"] == {"checkpoint_file": str(Path("run-1/latest.pth"))}


def test_build_resume_config_rejects_a_non_increasing_target() -> None:
    config = {"train": {"target_elements": 10_240}}

    with pytest.raises(ValueError, match="must exceed"):
        build_resume_config(config, Path("run-1/latest.pth"), 10_240)
