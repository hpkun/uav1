"""Focused tests of the read-only representation diagnostic, no training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.stea_mappo.networks import MaskedEntityAttention, SpatioTemporalEntityAttentionActor
from tools.diagnose_stea_representation import (concentration_rows,summarize_concentration,
    uniform_context,one_step_counterfactuals,memory_input,sha256,checkpoint_diagnostic,save_json,rollout_batched)

ROOT=Path(__file__).resolve().parents[1]


def test_uniform_onehot_entropy_and_effective_count():
    weights=torch.tensor([[[.25,.25,.25,.25]],[[1.,0.,0.,0.]]])
    rows,k=concentration_rows(weights,torch.ones(2,4,dtype=torch.bool))
    assert rows[0,0,1].item()==pytest.approx(1.)
    assert rows[1,0,1].item()==pytest.approx(0.,abs=1e-8)
    assert rows[0,0,5].item()==pytest.approx(4.)
    assert rows[1,0,5].item()==pytest.approx(1.)
    assert rows[0,0,4].item()==pytest.approx(0.)


def test_k0_k1_excluded_from_concentration():
    weights=torch.tensor([[[0.,0.]],[[1.,0.]],[[.5,.5]]])
    valid=torch.tensor([[False,False],[True,False],[True,True]])
    rows,k=concentration_rows(weights,valid)
    result=summarize_concentration(rows.numpy(),k.numpy())
    assert result['k0_excluded']==result['k1_excluded']==result['eligible_decisions']==1
    assert result['decision_mean']['normalized_entropy']['mean']==pytest.approx(1.)
    assert result['decision_mean']['effective_entity_count']['mean']==pytest.approx(2.)


def test_uniform_uses_projected_valid_values_and_empty_zero():
    model=MaskedEntityAttention(4,2)
    embeddings=torch.randn(2,3,4)
    valid=torch.tensor([[True,False,True],[False,False,False]])
    actual=uniform_context(model,embeddings,valid)
    expected=model.wv(embeddings)[0,[0,2]].mean(0)
    torch.testing.assert_close(actual[0],expected)
    assert torch.count_nonzero(actual[1])==0
    changed=embeddings.clone();changed[~valid]=1e9
    torch.testing.assert_close(uniform_context(model,changed,valid),actual)


def test_one_step_does_not_mutate_hidden_and_matches_manual_zero():
    actor=SpatioTemporalEntityAttentionActor()
    obs=torch.randn(5,65);obs[:,7:].reshape(5,-1)  # five-agent slots
    # Set valid entity indicators in the actual 7+28+30 layout.
    obs[:,7:35].reshape(5,4,7)[...,-1]=1.
    obs[:,35:].reshape(5,5,6)[...,-1]=1.
    hidden=torch.randn(5,128);saved=hidden.clone();alive=torch.ones(5);start=torch.tensor(0.)
    learned=actor.distribution_step(obs,hidden,alive,start)[0].mean.tanh()
    uniform_diff,zero_diff=one_step_counterfactuals(actor,obs,hidden,alive,start,learned)
    manual=actor.distribution_step(obs,torch.zeros_like(hidden),alive,start)[0].mean.tanh()
    assert torch.equal(hidden,saved)
    torch.testing.assert_close(zero_diff,(learned-manual).abs())
    assert uniform_diff.shape==(5,3)


def test_memoryless_input_is_zero_every_step_not_just_episode_start():
    hidden=torch.ones(5,128)
    for _ in range(5):
        actual=memory_input(hidden,True)
        assert not actual.any() and hidden.all()
        hidden=actual+2.
    assert memory_input(hidden,False) is hidden


def test_hash_records_and_non_overwriting_output(tmp_path):
    source=tmp_path/'checkpoint.pt';source.write_bytes(b'known bytes')
    assert sha256(source)==hashlib.sha256(b'known bytes').hexdigest()
    output=tmp_path/'result.json';save_json(output,dict(sha=sha256(source)))
    save_json(output,dict(sha=sha256(source)))
    with pytest.raises(FileExistsError):save_json(output,dict(sha='wrong'))


def test_real_checkpoint_reproducibility_hash_and_formal_eval(tmp_path):
    assert torch.cuda.is_available(),'CUDA required'
    run=ROOT/'outputs/stea_mappo_v25_5v5_seed1_2m_mlpcritic'
    checkpoint=run/'checkpoint_2000000.pt'
    if not checkpoint.exists():pytest.skip('local completed experiment not available')
    first=checkpoint_diagnostic(run,checkpoint,65000000,1,'cuda',True)
    second=checkpoint_diagnostic(run,checkpoint,65000000,1,'cuda',True)
    assert first==second
    assert first['checkpoint_sha256']==sha256(checkpoint)
    assert first['normal']['sequence_replay_audit']['pre_update_ratio_max_abs_error']<1e-4
    from algorithm.stea_mappo.evaluation import evaluate_stea_mappo_checkpoint
    import yaml
    cfg=yaml.safe_load((run/'algorithm_config.yaml').read_text())
    env=yaml.safe_load((run/'env_config.yaml').read_text())
    with torch.backends.cudnn.flags(benchmark=False,deterministic=True):
        formal=evaluate_stea_mappo_checkpoint(checkpoint,cfg,env,'cuda',[65000000])
    for key,value in first['normal']['metrics'].items():assert value==formal[key]
    path=tmp_path/'diagnostic.json';save_json(path,first)
    assert json.loads(path.read_text())==first


def test_batched_checkpoint_diagnostics_reproducible_and_episode_independent():
    assert torch.cuda.is_available(),'CUDA required'
    run=ROOT/'outputs/stea_mappo_v25_5v5_seed1_2m_mlpcritic';checkpoint=run/'checkpoint_2000000.pt'
    if not checkpoint.exists():pytest.skip('local completed experiment unavailable')
    first=checkpoint_diagnostic(run,checkpoint,65000000,2,'cuda',True)
    second=checkpoint_diagnostic(run,checkpoint,65000000,2,'cuda',True)
    assert first==second
    assert [row['environment_seed'] for row in first['normal']['episodes']]==[65000000,65000001]
    assert first['normal']['sequence_replay_audit']['pre_update_ratio_max_abs_error']<1e-4
