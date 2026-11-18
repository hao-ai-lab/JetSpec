"""Reduce one serve read (e.g. a 128-prompt JetSpec++ run outside the Table 3 plan) with the Table 3 reducer
(rule A, steady rounds), without the plan receipt checks.

usage: python bench/jetspec_plus_plus/dspark_opt/reduce_single.py read.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bench.jetspec_plus_plus.paper import command  # noqa: E402

output = Path(sys.argv[1])
campaign = json.loads(Path(str(output) + '.campaign.json').read_text())
raw = json.loads(output.read_text())
measured = campaign['reads'][0]['measured']
populations = json.loads(Path(str(output) + '.populations.json').read_text())['populations']
condition = campaign['resolved_condition']
rows = command.per_prompt(condition['method'], raw['outputs'], measured['metrics'], populations,
                          list(range(len(raw['outputs']))), 2048, condition['eos_token_ids'],
                          chain_lite=measured.get('chain', {}).get('anatomy_lite'))
values = command.reduce_rows(rows)['values']
values['exposed'] = values['round'] - values['verify']
print(json.dumps(values))
