from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tools.audit_dawe_fixed_bank_kl_decomposition import (
    BANK_PATH, MULTIPLIERS, directional_diagonal_gaussian_kl_components,
    effective_log_std, actor_outputs, contribution_fraction, final_path,
    safe_ratio, source_path, symmetric_diagonal_gaussian_kl_components,
    validate_checkpoint_identity, validate_formal_results,
)
from tools.fixed10_wave_state_drift_common import (
    checkpoint_mutation_guard, load_checkpoint,
    symmetric_diagonal_gaussian_kl, validate_bank,
)

ROOT = Path(__file__).resolve().parents[1]


def arrays():
    mu0=np.array([[0.1,-0.2,0.3],[0.4,0.0,-0.1]])
    log0=np.log(np.array([[0.8,1.0,1.2],[0.7,0.9,1.1]]))
    return mu0,log0


def test_identical_gaussian_all_components_zero():
    mu,log=arrays()
    for helper in (directional_diagonal_gaussian_kl_components,symmetric_diagonal_gaussian_kl_components):
        result=helper(mu,log,mu,log)
        assert np.allclose(result["total"],0) and np.allclose(result["mean"],0) and np.allclose(result["variance"],0)


def test_mean_only_directional_and_symmetric():
    mu,log=arrays();changed=mu+0.2
    for helper in (directional_diagonal_gaussian_kl_components,symmetric_diagonal_gaussian_kl_components):
        result=helper(mu,log,changed,log)
        assert np.allclose(result["variance"],0) and np.allclose(result["total"],result["mean"])


def test_variance_only_directional_and_symmetric():
    mu,log=arrays();changed=log+0.3
    for helper in (directional_diagonal_gaussian_kl_components,symmetric_diagonal_gaussian_kl_components):
        result=helper(mu,log,mu,changed)
        assert np.allclose(result["mean"],0) and np.allclose(result["total"],result["variance"])


@pytest.mark.parametrize("helper",[directional_diagonal_gaussian_kl_components,symmetric_diagonal_gaussian_kl_components])
def test_decomposition_identity(helper):
    mu,log=arrays();result=helper(mu,log,mu+0.17,log-0.22)
    assert np.allclose(result["total"],result["mean"]+result["variance"],atol=1e-12)


def test_symmetric_total_matches_existing_helper():
    mu,log=arrays();changed_mu=mu+0.17;changed_log=log-0.22
    result=symmetric_diagonal_gaussian_kl_components(mu,log,changed_mu,changed_log)
    assert np.allclose(result["total"],symmetric_diagonal_gaussian_kl(mu,log,changed_mu,changed_log),atol=1e-12)


@pytest.mark.parametrize("rho",[0.25,1.0])
def test_common_sigma_scaling_preserves_variance_component(rho):
    mu,log=arrays();changed_mu=mu+0.17;changed_log=log-0.22;shift=np.log(rho)
    for helper in (directional_diagonal_gaussian_kl_components,symmetric_diagonal_gaussian_kl_components):
        base=helper(mu,log,changed_mu,changed_log);effective=helper(mu,log+shift,changed_mu,changed_log+shift)
        assert np.allclose(base["variance"],effective["variance"],atol=1e-12)


@pytest.mark.parametrize("helper",[directional_diagonal_gaussian_kl_components,symmetric_diagonal_gaussian_kl_components])
def test_rho_quarter_amplifies_mean_component_sixteen_times(helper):
    mu,log=arrays();changed_mu=mu+0.17;changed_log=log-0.22
    base=helper(mu,log,changed_mu,changed_log);effective=helper(mu,log+np.log(.25),changed_mu,changed_log+np.log(.25))
    assert np.allclose(effective["mean"],16*base["mean"],rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize("helper",[directional_diagonal_gaussian_kl_components,symmetric_diagonal_gaussian_kl_components])
def test_rho_one_is_exact_identity(helper):
    mu,log=arrays();base=helper(mu,log,mu+.17,log-.22);effective=helper(mu,log+np.log(1.),mu+.17,log-.22+np.log(1.))
    for key in ("mean","variance","total"):assert np.array_equal(base[key],effective[key])


def test_registered_wave_multipliers_and_group_routing():
    assert MULTIPLIERS=={"Control":{1:1.,2:1.,3:1.},"DAWE":{1:.25,2:.25,3:1.}}
    log=np.zeros((3,4,3));wave=np.array([1,2,3]);effective=effective_log_std(log,wave,"DAWE")
    assert np.allclose(np.exp(effective[0]),.25) and np.allclose(np.exp(effective[1]),.25) and np.allclose(np.exp(effective[2]),1.)


def test_alive_mask_excludes_dead_agents_and_wave_groups_are_disjoint():
    with np.load(BANK_PATH,allow_pickle=False) as bank:
        observations,alive,wave=bank["observations"],bank["alive"],bank["wave"]
    validate_bank(observations,alive,wave)
    assert sum(int((wave==value).sum()) for value in (1,2,3))==len(wave)
    assert int((alive>0.5).sum())+int((alive<=0.5).sum())==alive.size
    assert set(np.unique(wave))=={1,2,3}


def test_fixed_bank_schema_is_exact():
    with np.load(BANK_PATH,allow_pickle=False) as bank:
        validate_bank(bank["observations"],bank["alive"],bank["wave"])
        assert bank["observations"].shape==(1000,4,52) and bank["alive"].shape==(1000,4) and bank["wave"].shape==(1000,)


def test_source_control_dawe_checkpoint_steps_and_seeds():
    for seed in (5301,5302,5303):
        source=load_checkpoint(source_path(seed),"cpu");validate_checkpoint_identity(source,1_505_280,seed,"source")
        for arm in ("Control","DAWE"):
            final=load_checkpoint(final_path(arm,seed),"cpu");validate_checkpoint_identity(final,1_805_280,seed,arm)


def test_formal_analysis_is_safety_fail_and_protocol_passes():
    analysis,protocol=validate_formal_results()
    assert analysis["DAWE_SCREEN"]=="SAFETY_FAIL" and analysis["uses_45m"] is False
    assert len(protocol)==6 and all(row["status"]=="PASS" for row in protocol.values())


def test_45m_guard_is_explicit_and_false_in_formal_analysis():
    analysis=json.loads((ROOT/"outputs/dawe_fixed10_300k_analysis/analysis.json").read_text())
    assert analysis["uses_45m"] is False
    source=(ROOT/"tools/audit_dawe_fixed_bank_kl_decomposition.py").read_text()
    assert '"accessed_45m": False' in source and '"used_44m": False' in source


def test_checkpoint_file_mutation_guard_is_stable():
    paths=[source_path(5301),final_path("Control",5301),final_path("DAWE",5301)]
    assert checkpoint_mutation_guard(paths)==checkpoint_mutation_guard(paths)


def test_actor_no_grad_forward_does_not_mutate_parameters():
    seed=5301;state=load_checkpoint(source_path(seed),"cpu")
    config=yaml.safe_load((ROOT/f"outputs/dev_dawe_control_seed{seed}_300k/algorithm_config.yaml").read_text())
    with np.load(BANK_PATH,allow_pickle=False) as bank:
        observations=np.asarray(bank["observations"][:4],dtype=np.float32);alive=np.asarray(bank["alive"][:4],dtype=np.float32)
    output,guard=actor_outputs(config,state,"cpu",observations,alive)
    assert guard["status"]=="PASS" and guard["before"]==guard["after"] and output["mu"].shape==(4,4,3)


def test_audit_source_has_no_environment_or_update_execution():
    source=(ROOT/"tools/audit_dawe_fixed_bank_kl_decomposition.py").read_text()
    assert ".step(" not in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "rsample(" not in source and ".sample(" not in source


def test_safe_ratio_near_zero_is_undefined():
    assert safe_ratio(1.,0.)=={"value":None,"status":"UNDEFINED"}
    assert safe_ratio(1.,1e-13)["status"]=="UNDEFINED"
    assert safe_ratio(2.,1.)=={"value":2.,"status":"DEFINED"}


def test_contribution_fraction_near_zero_is_undefined():
    assert contribution_fraction(0.,0.)=={"value":None,"status":"UNDEFINED"}
    assert contribution_fraction(.3,1.)=={"value":.3,"status":"DEFINED"}
