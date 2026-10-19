"""Sidecar diagnostics; canonical physical-action trace schema is unchanged."""
import json
from pathlib import Path
import numpy as np


def save_diagnostics(output,rows):
    output=Path(output)
    arrays={key:np.stack([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(output/'acirl_decisions.npz',**arrays)
    events=[];n=arrays['intent'].shape[1]
    for i in range(n):
        start=None
        for step,row in enumerate(rows):
            if row['gate'][i]:
                if start is not None:
                    events.append(dict(agent=i,start=start,end=step,duration=step-start,intention=int(rows[start]['intent'][i]),
                        end_reason=int(row['reason'][i]),selection_probability=float(rows[start]['intention_probability'][i])))
                start=step
            if start is not None and (row['episode_done'] or not row['next_alive'][i]):
                events.append(dict(agent=i,start=start,end=step+1,duration=step+1-start,intention=int(rows[start]['intent'][i]),
                    end_reason='episode_end' if row['episode_done'] else 'agent_death',selection_probability=float(rows[start]['intention_probability'][i])))
                start=None
    (output/'acirl_intention_events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    return dict(algorithm_decisions_file='acirl_decisions.npz',algorithm_intention_events_file='acirl_intention_events.json',
        intention_reason_codes={'0':'keep','1':'initialization','2':'max_duration','3':'learned_renew','4':'fixed_interval','5':'no_intent'},
        algorithm_decision_rows=len(rows))
