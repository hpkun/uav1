"""Read mixed historical and compact training logs without rewriting them."""
import json

def training_record_kind(row):
    kind = row.get('record_type')
    if kind is None:return 'legacy_step'
    if kind in ('ppo_update','rollout_summary'):return kind
    raise ValueError(f'unknown training record_type: {kind}')

def read_training_records(path):
    with open(path,encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row=json.loads(line)
                training_record_kind(row)
                yield row
