from copy import deepcopy
import hashlib
import json
import random

import numpy as np
import pytest
import torch
import yaml

from tools import diagnose_marc_deployment_policy as diagnostic


def metadata(method="marc_mappo_v2", version=2):
    return {"algorithm": "modular_mappo", "extra": {"development_method": method},
            "development_feature_versions": {"milestone_aware_retention_credit": version}}


@pytest.mark.parametrize("method,version", [("marc_mappo_v1", 1), ("marc_mappo_v2", 2)])
def test_valid_pairs(method, version):
    assert diagnostic.validate_method_version(metadata(method, version)) == (method, version)


@pytest.mark.parametrize("method,version", [("marc_mappo_v1", 2), ("marc_mappo_v2", 1),
                                           ("unknown", 2), (None, None)])
def test_invalid_pairs_explicit_error(method, version):
    with pytest.raises(RuntimeError, match="development_method=.*checkpoint MARC version=.*expected valid pair"):
        diagnostic.validate_method_version(metadata(method, version))


@pytest.fixture(params=[1, 2])
def checkpoint(request, tmp_path):
    assert torch.cuda.is_available(), "CUDA mandatory for checkpoint tests"
    version = request.param
    config = diagnostic.load_config(diagnostic.ROOT / f"configs/dev_marc_mappo_v{version}_1m.yaml")
    env = diagnostic.load_config(diagnostic.ROOT / "configs/persistent_wave_v2_blue433_environment.yaml")
    trainer = diagnostic.build_modular_mappo_trainer(config, "cuda")
    path = tmp_path / "best_eval.pt"
    state = metadata(f"marc_mappo_v{version}", version)
    state.update({"actor": trainer.actor.state_dict(), "sampled_steps": 123456,
                  "modular_mappo_impl_version": diagnostic.MODULAR_MAPPO_IMPL_VERSION,
                  "baseline_mappo_impl_version": diagnostic.MAPPO_IMPL_VERSION,
                  **trainer.module_protocol(),
                  "milestone_aware_retention_credit_state": {"elite_segments": "must not be used"},
                  "actor_optimizer": "must not be restored", "critic_optimizer": "must not be restored"})
    state["extra"].update({"training_seed": 5301, "environment_variant": "persistent_wave_v2",
                           "observation_dim": 52, "action_dim": 3, "num_agents": 4,
                           "runtime_environment_config": env})
    torch.save(state, path)
    (tmp_path / "algorithm_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config, env, state


def test_actor_only_strict_restore_and_provenance(checkpoint, monkeypatch):
    path, _, env, state = checkpoint
    def forbidden(*args, **kwargs):
        raise AssertionError("must not restore optimizer or training state")
    monkeypatch.setattr(torch.optim.Adam, "load_state_dict", forbidden)
    cpu = torch.get_rng_state().clone()
    cuda = [x.clone() for x in torch.cuda.get_rng_state_all()]
    trainer, report = diagnostic.load_deployment_checkpoint(path, env)
    assert torch.equal(cpu, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    assert all(torch.equal(trainer.actor.state_dict()[k], v) for k, v in state["actor"].items())
    assert report["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["checkpoint"] == str(path.resolve())
    assert report["marc_version"] in (1, 2)
    assert report["development_method"] == f"marc_mappo_v{report['marc_version']}"
    assert report["checkpoint_sampled_steps"] == 123456  # no hard-coded production step
    assert not report["optimizer_restored"] and not report["marc_module_state_restored"]
    assert trainer.actor_optimizer.state == {}
    assert trainer.milestone_aware_retention_credit.state_dict() != state["milestone_aware_retention_credit_state"]


@pytest.mark.parametrize("field,value", [("waves", 2), ("blue", [4, 4, 4]),
                                        ("max_steps", 1000), ("red", 3), ("variant", "other")])
def test_433_environment_mismatch(checkpoint, field, value):
    _, config, env, state = checkpoint
    changed = deepcopy(env)
    paths = {"waves": ("persistent_waves", "total_waves"), "blue": ("persistent_waves", "blue_units_per_wave"),
             "max_steps": ("simulation", "max_steps"), "red": ("scenario", "team_size")}
    if field == "variant": changed["environment_variant"] = value
    else:
        section, key = paths[field]
        changed[section][key] = value
    with pytest.raises(RuntimeError, match="433 environment identity mismatch"):
        diagnostic.validate_environment(state, config, changed)


@pytest.mark.parametrize("key,value", [("observation_dim", 53), ("action_dim", 4), ("num_agents", 3)])
def test_checkpoint_shapes_fail_fast(checkpoint, key, value):
    _, config, env, state = checkpoint
    state["extra"][key] = value
    with pytest.raises(RuntimeError, match=f"checkpoint {key} mismatch"):
        diagnostic.validate_environment(state, config, env)


def test_training_env_mismatch_rejected(checkpoint):
    _, config, env, state = checkpoint
    state["extra"]["runtime_environment_config"] = deepcopy(env)
    state["extra"]["runtime_environment_config"]["simulation"]["max_steps"] = 1000
    with pytest.raises(RuntimeError, match="checkpoint training"):
        diagnostic.validate_environment(state, config, env)


class TinyEnvironment:
    seen_seeds = []
    actions = []
    max_steps = 3000
    def reset(self, seed):
        self.seen_seeds.append(seed)
        self.steps = 0
        self.red_alive_mask = np.ones(4, np.float32)
        self.blue_alive_mask = np.ones(4, np.float32)
        self.obs = np.full((4, 52), .1, np.float32)
        return self.obs, {}
    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        info = {"red_alive_mask": self.red_alive_mask, "wave_index": 1,
                "waves_cleared": 0, "red_boundary_exits": 0, "red_ground_losses": 0,
                "episode_length": self.steps}
        return self.obs, np.full(4, action.mean(), np.float32), self.steps == 2, False, info


def test_real_actor_action_semantics_rng_and_elite_ignored(checkpoint, monkeypatch):
    path, _, env, _ = checkpoint
    trainer, _ = diagnostic.load_deployment_checkpoint(path, env)
    monkeypatch.setattr(diagnostic, "make_combat_environment", lambda config: TinyEnvironment())
    def forbidden(*args, **kwargs): raise AssertionError("elite retention must not run at deployment")
    monkeypatch.setattr(trainer.milestone_aware_retention_credit, "retention_loss", forbidden)
    obs = torch.full((1, 4, 52), .1, device="cuda")
    with torch.no_grad(): distribution = trainer.actor.distribution(obs)
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state().clone()
    TinyEnvironment.actions = []
    diagnostic.one_episode(trainer, env, 49100000, True, 123)
    np.testing.assert_array_equal(TinyEnvironment.actions[0], distribution.loc.tanh().cpu().numpy()[0])
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state())
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        expected = distribution.rsample().tanh().cpu().numpy()[0]
    TinyEnvironment.actions = []
    diagnostic.one_episode(trainer, env, 49100000, False, 123)
    np.testing.assert_array_equal(TinyEnvironment.actions[0], expected)
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state())


def test_repeat_seed_bank_reproducibility_and_report(checkpoint, monkeypatch):
    path, _, env, _ = checkpoint
    trainer, metadata = diagnostic.load_deployment_checkpoint(path, env)
    monkeypatch.setattr(diagnostic, "make_combat_environment", lambda config: TinyEnvironment())
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state().clone()
    numpy = np.random.get_state()
    python = random.getstate()
    TinyEnvironment.seen_seeds = []
    kwargs = dict(trainer=trainer, metadata=metadata, env_config=env, env_path="env.yaml",
                  episodes=2, seed_base=49100000, policy_mode="stochastic", stochastic_repeats=3)
    first = diagnostic.evaluate_deployment(**kwargs)
    assert TinyEnvironment.seen_seeds == [49100000, 49100001] * 3
    assert first == diagnostic.evaluate_deployment(**kwargs)
    assert len({r["episode_policy_seeds"][0] for r in first["per_repeat"]}) == 3
    assert all(r["environment_seed_range"] == [49100000,49100001] for r in first["per_repeat"])
    assert first["aggregate"]["Return"]["std"] > 0
    assert first["per_repeat"][0]["metrics"]["Q2"] is None
    assert first["per_repeat"][0]["metrics"]["Q3"] is None
    assert first["checkpoint_sha256"] and first["development_method"] and first["marc_version"]
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state())
    assert python == random.getstate()
    current = np.random.get_state()
    assert numpy[0] == current[0] and np.array_equal(numpy[1], current[1]) and numpy[2:] == current[2:]
    json.dumps(first, allow_nan=False)


def test_reserved_final_bank_rejected():
    with pytest.raises(ValueError, match="reserved"):
        diagnostic.evaluation_seed_bank(45000000, 1)
    with pytest.raises(ValueError, match="reserved"):
        diagnostic.evaluation_seed_bank(44999999, 2)


def test_no_cuda_fails_explicitly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is mandatory"):
        diagnostic.load_deployment_checkpoint("unused.pt", {})
