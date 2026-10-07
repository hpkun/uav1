from pathlib import Path
import math
from tools.audit_clipping_geometry import (clip_scale_proxy,wsai_post_pre_proxy,window_for_step,
 exact_intersection,paired_geometry,plain_status,excess_status,collapse_alignment,metric_or_na)

ROOT=Path(__file__).resolve().parents[1]

def test_plain_common_clip_proxy_formula():
 assert clip_scale_proxy(2.)==.25
 assert clip_scale_proxy(.25)==1.

def test_wsmh_common_proxy_is_not_exact_mean_scale():
 proxy=clip_scale_proxy(2.)
 exact_mean=(clip_scale_proxy(1.)+clip_scale_proxy(3.))/2
 assert proxy==.25 and exact_mean!=proxy

def test_wsai_post_pre_proxy_formula_and_zero_safety():
 assert wsai_post_pre_proxy(.5,2.)==.25
 assert math.isfinite(wsai_post_pre_proxy(0.,0.))

def test_window_boundaries_are_strict_lower_inclusive_upper():
 assert window_for_step(1_505_280) is None
 assert window_for_step(1_505_281)=="EARLY"
 assert window_for_step(1_603_584)=="EARLY"
 assert window_for_step(1_603_585)=="MIDDLE"
 assert window_for_step(1_805_280)=="LATE"

def test_exact_step_intersection_never_nearest_matches():
 assert exact_intersection([{1,2,3},{2,3,4},{3,4,5}])=={3}
 assert 2 not in exact_intersection([{2},{3},{2}])

def test_paired_preclip_and_scale_ratios():
 value=paired_geometry(2.,1.)
 assert value["preclip_ratio"]==2.
 assert value["scale_ratio"]==.5
 assert value["clip_severity_ratio"]==2.

def _plain(pressure,scale):return {str(s):{"clip_pressure_fraction":pressure[i],"clip_scale_proxy":{"median":scale[i]}} for i,s in enumerate((5301,5302,5303))}
def _paired(pre,scale):return {str(s):{"median_preclip_ratio":pre[i],"median_scale_ratio":scale[i]} for i,s in enumerate((5301,5302,5303))}

def test_plain_strong_mixed_weak_rules():
 assert plain_status(_plain([.9,.8,.2],[.2,.25,.9]))[0]=="STRONG"
 assert plain_status(_plain([.9,.2,.2],[.2,.9,.9]))[0]=="MIXED"
 assert plain_status(_plain([.2,.2,.2],[.9,.9,.9]))[0]=="WEAK"

def test_excess_clipping_rules():
 assert excess_status(_paired([1.3,1.4,1.],[.7,.8,1.]))[0]=="STRONG"
 assert excess_status(_paired([1.3,1.,1.],[.7,1.,1.]))[0]=="MIXED"
 assert excess_status(_paired([1.,1.,1.],[1.,1.,1.]))[0]=="WEAK"

def test_collapse_alignment_and_sparse_tail_rule():
 pre={"exact_scale_median":.4,"preclip_median":1.,"w1_later_median":1.,"backbone_median":1.,"kl_gt_005":0.}
 later={"exact_scale_median":.3,"preclip_median":1.3,"w1_later_median":1.3,"backbone_median":1.,"kl_gt_005":0.}
 assert collapse_alignment(pre,later,3)["label"]=="YES"
 assert collapse_alignment(pre,later,1)["label"]=="MIXED"

def test_missing_metric_reports_na():
 assert metric_or_na([{"x":1}],"missing")=="N/A"

def test_audit_source_has_no_runtime_or_mutating_calls_and_no_45m():
 source=(ROOT/"tools/audit_clipping_geometry.py").read_text(encoding="utf-8")
 forbidden=("evaluate_"+"modular","make_"+"combat_environment","optimizer"+".step","."+"backward(","trainer"+".update(","collect_"+"rollout","45000000","45000199")
 assert not [token for token in forbidden if token in source]
