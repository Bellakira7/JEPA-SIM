"""Protocol metadata and paired aggregation; independent of GPU dependencies."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = 'OpenScene-LTF-OL-v1'
AR_PROTOCOL = 'OpenScene-LTF-AR-v2'
FLOWSIM_COMMIT = '98f794c9055da0b55055e70e234d7785c0d2b88f'
LTF_REVISION = 'a068068540abdc32dd6985830ae8463a7ac2f9d2'
FLOWSIM_REVISION = 'cdb11ff7a62d8d21094ac0993034d2bed2956805'
OPENSCENE_REVISION = 'a76f840b65e972bc45e56c2adced897498e9a026'


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def paired_summary(records, arms, gaps, dataset, protocol=PROTOCOL):
    """Reject incomplete paired sets. Report token and log means separately."""
    arms, gaps = tuple(arms), tuple(gaps)
    if not arms or len(set(arms)) != len(arms) or len(set(gaps)) != len(gaps):
        raise ValueError('Unique nonempty arms and unique gaps are required')
    by_token = defaultdict(dict)
    logs = {}
    for r in records:
        token, key = r['token'], (r['arm'], r['gap_frames'])
        if key in by_token[token]:
            raise ValueError(f'Duplicate record: {token} {key}')
        if token in logs and logs[token] != r['log_name']:
            raise ValueError(f'Conflicting log for {token}')
        by_token[token][key] = r
        logs[token] = r['log_name']
    expected = {(a, g) for a in arms for g in gaps}
    if not by_token:
        raise ValueError('No paired records; empty evaluations cannot succeed')
    for token, group in by_token.items():
        if set(group) != expected:
            raise ValueError(f'Incomplete paired set for {token}')
    table = []
    for arm in arms:
        for gap in gaps:
            rows = [group[arm, gap] for group in by_token.values()]
            metric_names = set(rows[0]['metrics'])
            if any(set(r['metrics']) != metric_names for r in rows):
                raise ValueError('Metric schema changed across tokens')
            per_token, per_log, finite_counts = {}, {}, {}
            for metric in sorted(metric_names):
                vals_by_log = defaultdict(list)
                for r in rows:
                    value = r['metrics'][metric]
                    if value is not None and math.isfinite(value):
                        vals_by_log[r['log_name']].append(value)
                vals = [v for group in vals_by_log.values() for v in group]
                finite_counts[metric] = len(vals)
                per_token[metric] = sum(vals)/len(vals) if vals else None
                log_means = [sum(v)/len(v) for v in vals_by_log.values()]
                per_log[metric] = sum(log_means)/len(log_means) if log_means else None
            table.append(dict(arm=arm, gap_frames=gap, gap_seconds=gap*0.5,
                              per_token_mean=per_token, per_log_macro=per_log,
                              finite_token_counts=finite_counts))
    return dict(protocol=protocol, dataset=dataset,
                aggregation=['per_token_mean', 'per_log_macro'],
                num_tokens=len(by_token), num_logs=len(set(logs.values())), table=table)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)
