import json
import pytest
from tools.analyze_caiw_mappo import exact_row,nearest,primary_endpoint,primary_gate

def test_exact_three_million_endpoint_and_nearest():
    values=[{"sampled_steps":"2900000"},{"sampled_steps":"3000000"},{"sampled_steps":"3100000"}]
    assert exact_row(values,3000000)["sampled_steps"]=="3000000"
    assert nearest(values,2950000)["sampled_steps"] in {"2900000","3000000"}
    with pytest.raises(RuntimeError,match="CAIW_PRIMARY_ENDPOINT_3M_MISSING"):exact_row([values[0],values[2]],3000000)

def test_primary_endpoint_requires_summary_and_exact_history(tmp_path):
    run=tmp_path/"run";run.mkdir();(run/"run_summary.json").write_text(json.dumps({"sampled_steps":3000000}))
    assert primary_endpoint(run,[{"sampled_steps":"3000000"}])["sampled_steps"]=="3000000"
    with pytest.raises(RuntimeError,match="CAIW_DEVELOPMENT_INCOMPLETE"):
        primary_endpoint(run,[{"sampled_steps":"2900000"},{"sampled_steps":"3100000"}])
    (run/"run_summary.json").write_text(json.dumps({"sampled_steps":2900000}))
    with pytest.raises(RuntimeError,match="CAIW_DEVELOPMENT_INCOMPLETE"):
        primary_endpoint(run,[{"sampled_steps":"3000000"}])

def test_primary_gate_pass_and_single_failure():
    base={"delta_average_waves":.2,"delta_W1":0.,"delta_Q2":.1,"delta_Q3":.02}
    passed,means=primary_gate([{**base,"seed":s,"delta_average_waves":.2} for s in (1,2,3)])
    assert passed and means["delta_Q2"]>.05
    failed,_=primary_gate([{**base,"seed":1},{**base,"seed":2},{**base,"seed":3,"delta_Q2":-.1}])
    assert not failed
