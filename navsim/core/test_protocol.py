import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from protocol import paired_summary


def rows():
    return [dict(token=t, log_name=l, arm=a, gap_frames=g,
                 metrics={'planning/ade':v})
            for t,l,v in [('t1','l1',0.0),('t2','l1',2.0),('t3','l2',8.0)]
            for a in ['identity','copy_last_bev'] for g in [1,3]]


def test_log_and_token_aggregation_differ():
    r=paired_summary(rows(), ['identity','copy_last_bev'],[1,3],'mini')
    assert r['num_tokens']==3 and r['num_logs']==2
    assert r['table'][0]['per_token_mean']['planning/ade']==pytest.approx(10/3)
    assert r['table'][0]['per_log_macro']['planning/ade']==4.5


def test_incomplete_and_duplicates_fail():
    with pytest.raises(ValueError,match='Incomplete'):
        paired_summary(rows()[:-1],['identity','copy_last_bev'],[1,3],'mini')
    with pytest.raises(ValueError,match='Duplicate'):
        paired_summary(rows()+rows()[:1],['identity','copy_last_bev'],[1,3],'mini')


def test_empty_eval_fails():
    with pytest.raises(ValueError,match='No paired'):
        paired_summary([],['identity'],[1],'mini')


def test_nan_metrics_keep_sample_counts():
    r=rows()
    r[0]['metrics']['planning/ade']=math.nan
    s=paired_summary(r,['identity','copy_last_bev'],[1,3],'mini')
    assert s['table'][0]['finite_token_counts']['planning/ade']==2
    assert s['table'][0]['per_token_mean']['planning/ade']==5.0
