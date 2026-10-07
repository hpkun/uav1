from copy import deepcopy
from pathlib import Path
import pytest

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.train_modular_mappo import load_config
from tools.preflight_wave_balanced_curriculum import validate

ROOT = Path(__file__).resolve().parents[1]


def test_four_way_module_factorial_and_frozen_protocol():
    result = validate()
    assert result["status"] == "READY_FOR_WAVE_BALANCED_CURRICULUM_DEVELOPMENT"
    assert result["configs"]["wb_only"]["enabled_modules"] == ["actor_lr_decay", "wave_balancing"]
    assert result["configs"]["wec_only"]["enabled_modules"] == ["actor_lr_decay", "wave_entry_curriculum"]
    assert result["configs"]["proposed"]["enabled_modules"] == ["actor_lr_decay", "wave_balancing", "wave_entry_curriculum"]
    assert result["checks"]["future_45m_untouched"]


def test_old_and_new_curricula_are_mutually_exclusive():
    config = load_config(ROOT / "configs/dev_wave_entry_curriculum_3m.yaml")
    config = deepcopy(config); config["modules"]["curriculum"]["enabled"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_modular_mappo_trainer(config, "cpu", 16, 10)


def test_plain_path_has_no_wave_entry_curriculum():
    config = load_config(ROOT / "configs/diag_mappo_learnability_common_3m.yaml")
    trainer = build_modular_mappo_trainer(config, "cpu", 16, 10)
    assert not trainer.wave_entry_curriculum.enabled
    assert trainer.module_protocol()["enabled_modules"] == ["actor_lr_decay"]
